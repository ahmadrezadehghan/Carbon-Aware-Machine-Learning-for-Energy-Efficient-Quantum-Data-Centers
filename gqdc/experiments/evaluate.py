# Hook to compare different schedulers consistently (extend as needed)
import argparse, yaml
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheduler', choices=['fifo','milp','rl_ppo'], default='fifo')
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()
    print("Evaluation placeholder for", args.scheduler, "using", args.config)
if __name__ == '__main__':
    main()
