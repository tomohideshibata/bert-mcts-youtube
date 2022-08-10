import cshogi


class BasePlayer:
    def __init__(self):
        self.board = cshogi.Board()

    def usi(self):
        print('id name bert_player')
        print('usiok')

    def usinewgame(self):
        pass

    def setoption(self, option):
        pass

    def isready(self):
        pass

    def position(self, moves):
        if moves[0] == 'startpos':
            self.board.reset()
            for move in moves[2:]:
                self.board.push_usi(move)
        elif moves[0] == 'sfen':
            self.board.set_sfen(' '.join(moves[1:]))
        # for debug
        print(self.board.sfen())

    def set_limits(self, btime=None, wtime=None, byoyomi=None, binc=None, winc=None, nodes=None, infinite=False, ponder=False):
        pass

    def go(self):
        pass

    def ponderhit(self, last_limits):
        pass

    def quit(self):
        pass
