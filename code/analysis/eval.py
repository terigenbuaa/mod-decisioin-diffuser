if __name__ == '__main__':
    import os
    home_dir = os.path.expanduser("~")
    os.chdir(os.path.join(home_dir, "mod-decision-diffuser/code/analysis"))
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.evaluate_inv_parallel import evaluate
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep

    os.chdir(os.path.join(home_dir, "mod-decision-diffuser/code/analysis"))  # otherwise can't find default_inv.jsonl
    sweep = Sweep(RUN, Config).load("eval.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(evaluate, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
