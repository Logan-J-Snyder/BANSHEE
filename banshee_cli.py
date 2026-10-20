"""Installed entry point; preserve the standalone script's mode dispatch."""

import sys


def run():
    import BANSHEE_Lite_Chi as banshee

    if "--server" in sys.argv:
        sys.argv.remove("--server")
        banshee.server_main()
    else:
        banshee.main()


if __name__ == "__main__":
    run()
