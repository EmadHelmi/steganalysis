import argparse

ap = argparse.ArgumentParser()

ap.add_argument('-f', action='store_true', default=True)

args = ap.parse_args()
print(args.f)
