from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact, ttest_ind, ttest_ind_from_stats, norm, lognorm, t as tdist
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.api import OLS, WLS
from numba import jit
from numpy.random import default_rng, SeedSequence
from functools import partial
from scipy.optimize import root_scalar, minimize_scalar
import matplotlib.pyplot as plt
import warnings
from time import perf_counter
from outliers import adjbox, logbox
from randomized_control import RCT
import randomized_control
import multiprocessing


df, image = None, None
def test_plot(args):
    df, treat_col, target_col, control_value, exclude_tests = args
    rct = RCT(df, treat_col, target_col,
              outliers=target_col,
              exclude_tests=exclude_tests,
              control_val=bool(control_value))
    rct.eval_treat(verbose=0)
    incs = rct.results['incs']
    rct.plot()
    return incs.loc[pd.IndexSlice[:,'pv',:], :]['total'].to_list()

def power_plot(args):
    total_pop, ctrl_grp, inc_prop, mean, std_dev, alpha = args
    randomized_control.pre_exp_tt_plot_incpwr(total_pop, ctrl_grp, inc_prop, mean, std=std_dev, alpha=alpha, ret_values=False)
    return '1'


app = Flask(__name__, template_folder='templates', static_folder='/Users/sumedh/PycharmProjects/OMM_project/static')
@app.route('/')
def start():
    return render_template('start.html')


@app.route('/pre_experiment', methods=['GET', 'POST'])
def pre_exp():
    return render_template('pre_exp.html')

@app.route('/binary_1', methods=['GET', 'POST'])
def binary_1():
    return render_template('binary_1.html')

@app.route('/continuous_1', methods=['GET', 'POST'])
def continuous_1():
    return render_template('continuous_1.html')

@app.route('/binary_2', methods=['GET', 'POST'])
def binary_2():
    return render_template('binary_2.html')

@app.route('/continuous_2', methods=['GET', 'POST'])
def continuous_2():
    return render_template('continuous_2.html')


@app.route('/pre_exp_result', methods=['GET', 'POST'])
def pre_exp_result():
    mean = float(request.args.get('mean'))
    try:
        std_dev = float(request.args.get('std dev'))
    except:
        std_dev = None
    try:
        total_pop = int(request.args.get('total_pop'))
        ctrl_grp = None
    except:
        total_pop = None
        ctrl_grp = float(request.args.get('control'))
        ctrl_grp = ctrl_grp / 100
    inc_prop = float(request.args.get('inc'))
    inc_prop = inc_prop / 100
    alpha = float(request.args.get('alpha'))
    power = float(request.args.get('power'))
    pool = multiprocessing.Pool(processes=1)
    if total_pop == None:
        total_pop = randomized_control.pre_exp_tt_totalpop(ctrl_grp, inc_prop, mean, std=std_dev, alpha=alpha, power=power)
        effect_size, power_percent = randomized_control.pre_exp_tt_pwr(total_pop, ctrl_grp, inc_prop, mean, std=std_dev, alpha=alpha)
        check = pool.map(power_plot, [(total_pop, ctrl_grp, inc_prop, mean, std_dev, alpha)])[0]
        return render_template('pre_exp_result_1.html', total_pop=total_pop, alpha=alpha, power=power, effect_size=effect_size, power_percent=power_percent*100)
    elif ctrl_grp == None:
        ctrl_grp = randomized_control.pre_exp_tt_contprop(total_pop, inc_prop, mean, std= std_dev, alpha=alpha, power=power)
        effect_size, power_percent = randomized_control.pre_exp_tt_pwr(total_pop, ctrl_grp, inc_prop, mean, std=std_dev, alpha=alpha)
        check = pool.map(power_plot, [(total_pop, ctrl_grp, inc_prop, mean, std_dev, alpha)])[0]
        return render_template('pre_exp_result_2.html', ctrl_grp=ctrl_grp*100, alpha=alpha, power=power, effect_size=effect_size, power_percent=power_percent*100)


@app.route('/redirect')
def choose():
    option_1 = request.args.get('binary')
    option_2 = request.args.get('choice')
    choice = request.args.get('exp')
    if option_1 == "Binary"and option_2 == "Total Population":
        return "<script>window.location.replace('/binary_2')</script>", 200
    elif option_1 == "Binary" and option_2 == "Control Group %":
        return "<script>window.location.replace('/binary_1')</script>", 200
    elif option_1 == 'Continuous' and option_2 == "Total Population":
        return "<script>window.location.replace('/continuous_2')</script>", 200
    elif option_1 == 'Continuous' and option_2 == "Control Group %":
        return "<script>window.location.replace('/continuous_1')</script>", 200
    elif choice == 'Pre Experiment':
        return "<script>window.location.replace('/pre_experiment')</script>", 200
    elif choice == 'Post Experiment':
        return "<script>window.location.replace('/post_experiment')</script>", 200


@app.route('/post_experiment', methods=['GET', 'POST'])
def post_exp():
    return render_template('upload.html')


@app.route('/select_1', methods=['GET', 'POST'])
def select_1():
    if request.method == 'POST':
        file = request.files['inputFile']
        global df
        df = pd.read_csv(file)
        return render_template('dropdown_1.html', treatment=df.columns)


@app.route('/select_2', methods=['GET', 'POST'])
def select_2():
    if request.method =='POST':
        treat_col = request.form.get('treatment')
        v_1 = df[treat_col].value_counts(ascending=True).rename_axis('unique_values').reset_index(name='counts')['unique_values'][0]
        v_2 = df[treat_col].value_counts(ascending=True).rename_axis('unique_values').reset_index(name='counts')['unique_values'][1]
        control_list = [v_1, v_2]
        test_types = ['Welch T-Test', 'Bootstrap', 'Mannwhitney U-Test', 'Bayesian Test']
        return render_template('dropdown_2.html', treat_col=treat_col, target_col=df.columns, control_list=control_list, test_types=test_types)


@app.route('/post_exp_result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        treat_col = request.form.get('treatment')
        target_col = request.form.getlist("target")
        target_col.sort()
        get_test = request.form.getlist("test_types")
        all_tests = ['t', 'boo', 'u', 'bay']
        test_types = {'Welch T-Test':'t', 'Bootstrap':'boo', 'Mannwhitney U-Test':'u', 'Bayesian Test':'bay'}
        include_tests = [test_types[i] for i in get_test]
        include_tests.sort()
        exclude_test = [i for i in all_tests if i not in include_tests]
        control_value = request.form.get("control")
        pool = multiprocessing.Pool(processes=1)
        p_values = pool.map(test_plot, [(df, treat_col, target_col, control_value, exclude_test)])[0]

        return render_template('post_exp_result.html', p_test_tar=zip(p_values, include_tests*len(target_col), list(np.repeat(target_col, len(include_tests)))))



if __name__ == '__main__':
    app.run(host='192.168.29.80', port=1505, debug=True)