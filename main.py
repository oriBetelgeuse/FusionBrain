from argparse import ArgumentParser

from omegaconf import OmegaConf

from fb_baseline.train_detection_vqa import run_train

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='config path')
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    run_train(conf)
