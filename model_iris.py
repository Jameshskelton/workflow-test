from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd
import argparser
import sys

def make_prediction(model, data):
    x = model.predict(data.drop('species', axis = 0)
    df = pd.DataFrame(iris.data)
    df['actual'] = iris.target
    df['pred'] = x
    df.to_csv('predSpecies.csv')


def main(model, data):
    parser = argparse.ArgumentParser(
        description='''iris predictor. Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_gen = subparsers.add_parser('make_prediction', help='Generate images')
    parser_gen.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_gen.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'make_prediction': 'workflow-test.make_prediction'}
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)


if __name__ == "__main__":
	main()