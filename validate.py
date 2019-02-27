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
    def __init__(self, model, tgt_padding_idx):
        self.model = model
        self.tgt_padding_idx = tgt_padding_idx

        # Set model in training mode.
        self.model.eval()

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        """
        # Set model in validating mode.
        stats = {'support': 0, 'tgt_words': 0, 'src_words': 0, 'attended': 0, 'attended_possible': 0}
        with torch.no_grad():
            for batch in valid_iter:
                src = make_features(batch, 'src', 'text')
                _, src_lengths = batch.src
                stats['src_words'] += src_lengths.sum().item()

                tgt = make_features(batch, 'tgt')

                # F-prop through the model.
                outputs, attns = self.model(src, tgt, src_lengths)
                # outputs is seq x batch x hidden_size
                bottled_out = outputs.view(-1, outputs.size(-1))
                generator_out = self.model.generator(bottled_out)

                tgt_lengths = tgt[1:].squeeze(2).ne(self.tgt_padding_idx).sum(dim=0)
                grid_sizes = src_lengths * tgt_lengths
                stats['attended_possible'] += grid_sizes.sum().item()

                out_support = generator_out.gt(0).sum(dim=1)
                tgt_non_pad = tgt[1:].ne(self.tgt_padding_idx).view(-1)
                support_non_pad = out_support.masked_select(tgt_non_pad)
                tgt_words = support_non_pad.size(0)
                stats['support'] += support_non_pad.sum().item()
                stats['tgt_words'] += tgt_words

                attn = attns['std']
                attn = attn.view(-1, attn.size(-1))
                attended = attn.gt(0).sum(dim=1)
                attended_non_pad = attended.masked_select(tgt_non_pad)
                stats['attended'] += attended_non_pad.sum().item()
                '''
                print(src.size())
                print(tgt.size())
                foo = attns['std'].squeeze(1)
                print(foo.size())
                print(foo.sum())
                # what's going on here: the tgt is size 10, the src is size 8,
                # the attention is (9 x 8).
                print('attn nonzeros', foo.gt(0).sum().item())
                print('total src words', src_lengths.sum().item())
                print('total tgt words', tgt_lengths.sum().item())
                print('src seq', [self.fields['src'].vocab.itos[i] for i in src])
                print('tgt seq', [self.fields['tgt'].vocab.itos[i] for i in tgt])
                print(foo)
                '''

        return stats


def build_dataset_iter(datasets, fields, batch_size, use_gpu):
    device = "cuda" if use_gpu else "cpu"
    return DatasetLazyIter(datasets, fields, batch_size, None, device, False)


def load_model(checkpoint, fields, k=0, bisect_iter=0, gpu=False):
    model_opt = checkpoint['opt']
    alpha_lookup = {'softmax': 1.0, 'tsallis15': 1.5, 'sparsemax': 2.0}
    if not hasattr(model_opt, 'loss_alpha'):
        model_opt.loss_alpha = alpha_lookup[model_opt.generator_function]
    gen_alpha = alpha_lookup.get(model_opt.generator_function,
                                 model_opt.loss_alpha)
    if not hasattr(model_opt, 'global_attention_alpha'):
        model_opt.global_attention_alpha = alpha_lookup[model_opt.global_attention_function]
    if not hasattr(model_opt, 'global_attention_bisect_iter'):
        model_opt.global_attention_bisect_iter = 0
    model = build_base_model(model_opt, fields, gpu, checkpoint)

    assert opt.k == 0 or opt.bisect_iter == 0, \
        "Bisection and topk are mutually exclusive ! !"
    if gen_alpha == 1.0:
        gen_func = nn.Softmax(dim=-1)
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

        tgt_padding_idx = fields['tgt'].vocab.stoi[PAD_WORD]

        validator = Validator(model, tgt_padding_idx)
        if opt.verbose:
            print(model.decoder.attn)
            print(model.generator)

        # I hate that this has to load the data twice
        dataset = torch.load(opt.data + '.' + 'valid' + '.0.pt')

        def valid_iter_fct(): return build_dataset_iter(
            iter([dataset]), fields, opt.batch_size, opt.gpu)

        valid_stats = validator.validate(valid_iter_fct())
        print('avg. attended positions/tgt word: {}'.format(valid_stats['attended'] / valid_stats['tgt_words']))
        print('avg. support size: {}'.format(valid_stats['support'] / valid_stats['tgt_words']))
        print('attention density: {}'.format(valid_stats['attended'] / valid_stats['attended_possible']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-verbose', action='store_true')
    parser.add_argument('-gpu', action='store_true')
    parser.add_argument('-models', nargs='+')
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-k', default=0, type=int)
    parser.add_argument('-bisect_iter', default=0, type=int)
    opt = parser.parse_args()
    main(opt)
