"""Actor-only deployment evaluation; no Critic or optimizer is constructed."""
from experiment import main

if __name__ == '__main__':
    main('eval')
