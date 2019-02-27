#!/usr/bin/env python

import argparse

import torch
import torch.nn as nn

import onmt

from onmt.inputters.inputter import load_fields_from_vocab, make_features, \
    DatasetLazyIter
from onmt.inputters import PAD_WORD
from onmt.model_builder import build_base_model


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


def build_validator(model, fields):
    """

    Args:
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


def build_dataset_iter(datasets, fields, batch_size, use_gpu):
    device = "cuda" if use_gpu else "cpu"

    batch_size_fn = None
    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, False)


def load_model(checkpoint, fields, k=0, bisect_iter=0, gpu=False):


    model_opt = checkpoint['opt']
    alpha_lookup = {'softmax': 1.0, 'tsallis15': 1.5, 'sparsemax': 2.0}
    if not hasattr(model_opt, 'loss_alpha'):
        model_opt.loss_alpha = alpha_lookup[model_opt.generator_function]
    gen_alpha = alpha_lookup.get(model_opt.generator_function,
                                 model_opt.loss_alpha)
    if not hasattr(model_opt, 'global_attention_alpha'):
        model_opt.global_attention_alpha = alpha_lookup[model_opt.global_attention_function]
    model = build_base_model(model_opt, fields, gpu, checkpoint)

    assert opt.k == 0 or opt.bisect_iter == 0, \
        "Bisection and topk are mutually exclusive ! !"
    if gen_alpha == 1.0:
        gen_func = nn.LogSoftmax(dim=-1)
    elif gen_alpha == 2.0:
        if k > 0:
            gen_func = onmt.modules.sparse_activations.SparsemaxTopK(dim=-1, k=k)
        elif bisect_iter > 0:
            gen_func = onmt.modules.sparse_activations.SparsemaxBisect(n_iter=bisect_iter)
        else:
            gen_func = onmt.modules.sparse_activations.Sparsemax(dim=-1)
    elif gen_alpha == 1.5 and bisect_iter == 0:
        if k > 0:
            gen_func = onmt.modules.sparse_activations.Tsallis15TopK(dim=-1, k=k)
        else:
            gen_func = onmt.modules.sparse_activations.Tsallis15(dim=-1)
    else:
        # generic tsallis with bisection
        assert bisect_iter > 0, "Must use bisection with alpha != 1,1.5,2"
        gen_func = onmt.modules.sparse_activations.TsallisBisect(
            alpha=gen_alpha, n_iter=bisect_iter)

    gen_weights = model.generator[0] if \
        isinstance(model.generator, nn.Sequential) else model.generator

    generator = nn.Sequential(gen_weights, gen_func)
    model.generator = generator

    model.eval()
    model.generator.eval()

    return model


def main(opt):
    # Build model.
    for path in opt.models:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        fields = load_fields_from_vocab(checkpoint['vocab'], 'text')
        fields = {'src': fields['src'], 'tgt': fields['tgt']}
        model = load_model(
            checkpoint, fields, k=opt.k, bisect_iter=opt.bisect_iter, gpu=opt.gpu)

        validator = build_validator(model, fields)
        print(model.generator)

        # I hate that this has to load the data twice
        dataset = torch.load(opt.data + '.' + 'valid' + '.0.pt')

        def valid_iter_fct(): return build_dataset_iter(
            iter([dataset]), fields, opt.batch_size, opt.gpu)

        valid_stats = validator.validate(valid_iter_fct())
        # print('avg. attended positions/tgt word: {}'.format(valid_stats['attended_pos'] / valid_stats['tgt_words']))
        print('avg. support size: {}'.format(valid_stats['support'] / valid_stats['tgt_words']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-gpu', action='store_true')
    parser.add_argument('-models', nargs='+')
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-k', default=0, type=int)
    parser.add_argument('-bisect_iter', default=0, type=int)
    opt = parser.parse_args()
    main(opt)
