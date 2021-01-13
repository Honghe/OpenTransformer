import argparse
import logging
import time

import torch
import torchaudio
import yaml

from otrans.data import load_vocab
from otrans.data.audio import normalization
from otrans.model import End2EndModel
from otrans.recognize import build_recognizer
from otrans.train.utils import map_to_cuda

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main(args):
    checkpoint = torch.load(args.load_model, map_location=torch.device('cpu'))

    if args.config is not None:
        with open(args.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        params = checkpoint['params']

    params['data']['batch_size'] = args.batch_size
    model_type = params['model']['type']
    model = End2EndModel[model_type](params['model'])

    if 'frontend' in checkpoint:
        model.frontend.load_state_dict(checkpoint['frontend'])
        logger.info('[FrontEnd] Load the frontend checkpoint!')

    model.encoder.load_state_dict(checkpoint['encoder'])
    logger.info('[Encoder] Load the encoder checkpoint!')

    if 'decoder' in checkpoint:
        model.decoder.load_state_dict(checkpoint['decoder'])
        logger.info('[Decoder] Load the decoder checkpoint!')

    if 'joint' in checkpoint:
        model.joint.load_state_dict(checkpoint['joint'])
        logger.info('[JointNet] Load the joint net of transducer checkpoint!')

    if 'look_ahead_conv' in checkpoint:
        model.lookahead_conv.load_state_dict(checkpoint['look_ahead_conv'])
        logger.info('[LookAheadConvLayer] Load the external lookaheadconvlayer checkpoint!')

    if 'ctc' in checkpoint:
        model.assistor.load_state_dict(checkpoint['ctc'])
        logger.info('[CTC Assistor] Load the ctc assistor checkpoint!')
    logger.info('Finished! Loaded pre-trained model from %s' % args.load_model)

    model.eval()
    if args.ngpu > 0:
        model.cuda()

    unit2idx = load_vocab(params['data']['vocab'])
    idx2unit = {i: c for (c, i) in unit2idx.items()}
    recognizer = build_recognizer(model_type, model, None, args, idx2unit)

    # recog
    inputs = wav_loader(args.wav, params['data'])
    if args.ngpu > 0:
        inputs = map_to_cuda(inputs)

    enc_inputs = inputs['inputs']
    enc_mask = inputs['mask']

    st = time.time()
    preds, scores = recognizer.recognize(enc_inputs, enc_mask)
    et = time.time()
    print(f'time elapse: {et - st:.3}')
    print(preds[0])
    print(scores[0])


def wav_loader(path, params):
    wavform, sample_frequency = torchaudio.load_wav(path)

    feature = torchaudio.compliance.kaldi.fbank(
        wavform, num_mel_bins=params['num_mel_bins'],
        sample_frequency=sample_frequency, dither=0.0
    )

    if params['normalization']:
        feature = normalization(feature)

    feature_length = feature.shape[0]
    inputs = {
        'inputs': feature.unsqueeze(0),
        'mask': torch.BoolTensor([True] * feature_length).unsqueeze(0)
    }

    return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-nb', '--nbest', type=int, default=1)
    parser.add_argument('-bw', '--beam_width', type=int, default=5)
    parser.add_argument('-pn', '--penalty', type=float, default=0.6)
    parser.add_argument('-ld', '--lamda', type=float, default=5)
    parser.add_argument('-m', '--load_model', type=str, default=None)
    parser.add_argument('-lm', '--load_language_model', type=str, default=None)
    parser.add_argument('-ngram', '--ngram_lm', type=str, default=None)
    parser.add_argument('-alpha', '--alpha', type=float, default=0.1)
    parser.add_argument('-beta', '--beta', type=float, default=0.0)
    parser.add_argument('-lmw', '--lm_weight', type=float, default=0.1)
    parser.add_argument('-cw', '--ctc_weight', type=float, default=0.0)
    parser.add_argument('-d', '--decode_set', type=str, default='test')
    parser.add_argument('-ml', '--max_len', type=int, default=60)
    parser.add_argument('-md', '--mode', type=str, default='beam')
    # transducer related
    parser.add_argument('-mt', '--max_tokens_per_chunk', type=int, default=5)
    parser.add_argument('-pf', '--path_fusion', action='store_true', default=False)
    parser.add_argument('-s', '--suffix', type=str, default=None)
    parser.add_argument('-p2w', '--piece2word', action='store_true', default=False)
    parser.add_argument('-resc', '--apply_rescoring', action='store_true', default=False)
    parser.add_argument('-lm_resc', '--apply_lm_rescoring', action='store_true', default=False)
    parser.add_argument('-rw', '--rescore_weight', type=float, default=1.0)
    parser.add_argument('-debug', '--debug', action='store_true', default=False)
    parser.add_argument('-sba', '--sort_by_avg_score', action='store_true', default=False)
    parser.add_argument('-ns', '--num_sample', type=int, default=1)
    parser.add_argument('wav')
    cmd_args = parser.parse_args()

    main(cmd_args)
