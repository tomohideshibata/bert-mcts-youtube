from src.player.base_player import BasePlayer


def usi(player: BasePlayer):
    while True:
        cmd_line = input()
        cmd = cmd_line.split(' ', 1)
        cmd = [c.rstrip() for c in cmd]

        if cmd[0] == 'usi':
            player.usi()
        elif cmd[0] == 'setoption':
            option = cmd[1].split(' ')
            player.setoption(option)
        elif cmd[0] == 'isready':
            player.isready()
        elif cmd[0] == 'usinewgame':
            player.usinewgame()
        elif cmd[0] == 'position':
            moves = cmd[1].split(' ')
            player.position(moves)
        elif cmd[0] == 'go':
            kwargs = {}
            if len(cmd) > 1:
                args = cmd[1].split(' ')
                if args[0] == 'infinite':
                    kwargs['infinite'] = True
                else:
                    if args[0] == 'ponder':
                        kwargs['ponder'] = True
                        args = args[1:]
                    for i in range(0, len(args) - 1, 2):
                        if args[i] in ['btime', 'wtime', 'byoyomi', 'binc', 'winc', 'nodes']:
                            kwargs[args[i]] = int(args[i + 1])
            self.set_limits(**kwargs)

            player.go()
        elif cmd[0] == 'quit':
            player.quit()
            break
