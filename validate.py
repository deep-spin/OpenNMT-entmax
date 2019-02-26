#!/usr/bin/env python

import configargparse

import os
import random
import torch
import torch.nn as nn

import onmt
import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, make_features
from onmt.inputters import PAD_WORD
from onmt.model_builder import build_model
from onmt.model_builder import load_test_model
from onmt.utils.logging import init_logger, logger


class Validator(object):
    def __init__(self, model, padding_idx):
        self.model = model
        self.padding_idx = padding_idx

        # Set model in training mode.
        self.model.eval()

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        stats = {'support': 0, 'tgt_words': 0, 'attended_pos': 0}
        # what 
        with torch.no_grad():
            for batch in valid_iter:
                src = make_features(batch, 'src', 'text')
                _, src_lengths = batch.src

                tgt = make_features(batch, 'tgt')

                # F-prop through the model.
                outputs, attns = self.model(src, tgt, src_lengths)
                # outputs is seq x batch x hidden_size
                bottled_out = outputs.view(-1, outputs.size(-1))
                generator_out = self.model.generator(bottled_out)

                out_support = generator_out.gt(0).sum(dim=1)
                tgt_non_pad = tgt[1:].ne(self.padding_idx).view(-1)
                support_non_pad = out_support.masked_select(tgt_non_pad)
                stats['support'] += support_non_pad.sum().item()
                stats['tgt_words'] += support_non_pad.size(0)

                attn = attns['std']
                attn = attn.view(-1, attn.size(-1))
                attended = attn.ne(0).sum(dim=1)
                attended_non_pad = attended.masked_select(tgt_non_pad)
                stats['attended_pos'] += attended_non_pad.sum().item()

        return stats


def build_validator(opt, model, fields):
    """

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    padding_idx = fields['tgt'].vocab.stoi[PAD_WORD]

    return Validator(model, padding_idx)


def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size
        if opt.model_type == 'text' and opt.enc_rnn_size != opt.dec_rnn_size:
            raise AssertionError("""We do not support different encoder and
                                 decoder rnn sizes for translation now.""")

    opt.brnn = (opt.encoder_type == "brnn")

    if torch.cuda.is_available() and not opt.gpu_ranks:
        logger.info("WARNING: You have a CUDA device, \
                    should run with -gpu_ranks")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt):
    nb_gpu = len(opt.gpu_ranks)

    device_id = 1 if nb_gpu == 1 else -1

    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    assert opt.train_from is not None
    logger.info('Loading checkpoint from %s' % opt.train_from)
    checkpoint = torch.load(opt.train_from,
                            map_location=lambda storage, loc: storage)

    # Load default opts values then overwrite it with opts from
    # the checkpoint. It's usefull in order to re-train a model
    # after adding a new option (not set in checkpoint)
    dummy_parser = configargparse.ArgumentParser()
    opts.model_opts(dummy_parser)
    default_opt = dummy_parser.parse_known_args([])[0]

    model_opt = default_opt
    model_opt.__dict__.update(checkpoint['opt'].__dict__)

    # Peek the first dataset to determine the data_type.
    first_dataset = next(lazily_load_dataset("valid", opt))

    # Load fields generated from preprocess phase.
    fields = _load_fields(first_dataset, 'text', opt, checkpoint)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)

    alpha_lookup = {'softmax': 1.0, 'tsallis15': 1.5, 'sparsemax': 2.0}
    gen_alpha = alpha_lookup.get(model_opt.generator_function,
                                 model_opt.loss_alpha)
    assert opt.k == 0 or opt.bisect_iter == 0, \
        "Bisection and topk are mutually exclusive ! !"
    if gen_alpha == 1.0:
        gen_func = nn.LogSoftmax(dim=-1)
    elif gen_alpha == 2.0:
        if opt.k > 0:
            gen_func = onmt.modules.sparse_activations.SparsemaxTopK(dim=-1, k=opt.k)
        elif opt.bisect_iter > 0:
            gen_func = onmt.modules.sparse_activations.SparsemaxBisect(n_iter=opt.bisect_iter)
        else:
            gen_func = onmt.modules.sparse_activations.Sparsemax(dim=-1)
    elif gen_alpha == 1.5 and opt.bisect_iter == 0:
        if opt.k > 0:
            gen_func = onmt.modules.sparse_activations.Tsallis15TopK(dim=-1, k=opt.k)
        else:
            gen_func = onmt.modules.sparse_activations.Tsallis15(dim=-1)
    else:
        # generic tsallis with bisection
        assert opt.bisect_iter > 0, "Must use bisection with alpha != 1,1.5,2"
        gen_func = onmt.modules.sparse_activations.TsallisBisect(
            alpha=gen_alpha, n_iter=opt.bisect_iter)

    gen_weights = model.generator[0] if \
        isinstance(model.generator, nn.Sequential) else model.generator

    generator = nn.Sequential(gen_weights, gen_func)
    model.generator = generator

    model.eval()
    model.generator.eval()
    
    print(model)

    validator = build_validator(opt, model, fields)

    def train_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("train", opt), fields, opt)

    def valid_iter_fct(): return build_dataset_iter(
        lazily_load_dataset("valid", opt), fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Validating on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Validating on CPU, could be very slow')

    valid_stats = validator.validate(valid_iter_fct())
    print('avg. attended positions/tgt word: {}'.format(valid_stats['attended_pos'] / valid_stats['tgt_words']))
    print('avg. support size: {}'.format(valid_stats['support'] / valid_stats['tgt_words']))


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='train.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
