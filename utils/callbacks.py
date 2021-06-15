import re
from utils import camel2snake


class Callback():
    _order = 0
    def set_runner(self, run):
        self.run = run

    def __getattr__(self, k): 
        return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False


class TrainEvalCallback(Callback):
    def fit_start(self):
        pass

    def fit_end(self):
        pass

    def epoch_start(self):
        pass

    def epoch_end(self):
        pass

    def validate_start(self):
        pass

    def validate_end(self):
        pass

        
