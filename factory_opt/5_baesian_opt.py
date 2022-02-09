from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

class OPT:
    """ optimizer class """
    def __init__(self, objective_fun, bounds):
        self.optimizer = BayesianOptimization(f=objective_fun, 
                                                pbounds=bounds,
                                                verbose=2, random_state=1)

    def run(self, init_points=2, n_iter=3):
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        print(self.optimizer.max)

    def change_bounds(self, new_bounds):
        self.optimizer.set_bounds(new_bounds=new_bounds)

    def guide(self, params):
        self.optimizer.probe(params=params, lazy=True)

    def get_history(self):
        for i,res in enumerate(self.optimizer.res):
            print(f"Iteration {i}: \t {res}")

    def save(self, filename='factory_opt/log.json'):
        logger = JSONLogger(path=filename)
        self.optimizer.subscribe(event=Events.OPTIMIZATION_STEP, subscriber=logger)

    def load(self, filename='factory_opt/log.json'):
        load_logs(self.optimizer, logs=filename)
        print(f"{len(self.optimizer.space)} points loaded successfully..")

if __name__=="__main__":

    objective_fun = lambda x,y: -x**2-(y-1)**2+1
    bounds = {'x':(2,4), 'y':(-3,3)}

    # create optimizer, run, save and print
    Opt = OPT(objective_fun=objective_fun, bounds=bounds)
    Opt.save()
    Opt.run()
    Opt.get_history()

    # change bounds
    Opt.change_bounds(new_bounds={'x':(-2,3)})
    Opt.run()

    # create new object and load logs
    Opt1 = OPT(objective_fun=objective_fun, bounds=bounds)
    Opt1.load()

    