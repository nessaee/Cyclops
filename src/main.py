from analysis import Analysis
import argparse
from datetime import datetime
opt = ""
PROFILER = True

def create_directory(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

def main():
    global filename
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', action='store_true', help='Distance band sweep')
    parser.add_argument('--theta', action='store_true', help='')
    parser.add_argument('--kernel', action='store_true', help='')
    parser.add_argument('--x_px', action='store_true', help='')
    parser.add_argument('--claimed', action='store_true', help='')
    parser.add_argument('--initialize', action='store_true', help='')
    parser.add_argument('--dynamic', action='store_true', help='')
    parser.add_argument('--visible', action='store_true', help='')
    parser.add_argument('--iou', action='store_true', help='')
    parser.add_argument('--window', action='store_true', help='')
    parser.add_argument('--encode', action='store_false', help='Applies encoding before evaluation')
    parser.add_argument('--threshold', action='store_true', help='Similarity threshold sweep') 
    parser.add_argument('--scoring', action='store_true', help='Test proportion threshold sweep') 
    parser.add_argument('--rate', action='store_true', help='Sample rate sweep') 
    parser.add_argument('--sequence', action='store_true', help='')   
    parser.add_argument('--sim', action='store_true', help='') 
    parser.add_argument('-a', '--alpha', type=float, default=0, help='Similarity threshold ajustment from 0.5 (hundredths)')
    parser.add_argument('-b', '--beta', type=float, default=0, help='Beta')
    parser.add_argument('-t', '--tau', type=float, default=1, help='Tau')
    parser.add_argument('-K', '--K', type=int, default=1, help='Number of tests to be conducted')
    parser.add_argument('-k', '--k', type=int, default=1, help='Kernel size')
    parser.add_argument('-q', '--Q', type=int, default=2, help='Maximum state value')
    parser.add_argument('-s', '--s', type=int, default=30, help='Sample rate (images/second)')
    parser.add_argument('-ab', '--ab', action='store_true', help='Alpha and Beta sweep')
    parser.add_argument('-f', '--filter', action='store_true', help='Filters the dataset from designated .pkl')
    parser.add_argument('-r', '--regions', type=int, default=1, help='Number of regions for scene encoding')
    parser.add_argument('--save', type=str, default="default", help='Base directory containing CSV files')
    parser.add_argument('--process', action='store_true', help='')
    opt = parser.parse_args()

    filename = []
    for key,val in vars(opt).items():
        if type(val)==bool and val:
            filename.append(key)
        elif type(val)==int:
            filename.append(str((key,val)))

    filename = "-".join(filename)
    print(filename)
    obj = Analysis(**vars(opt))
    obj.analysis()

def profile_run():

    global filename
    import cProfile, pstats, io, re
    import pandas as pd
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    
    result = io.StringIO()
    pstats.Stats(pr,stream=result).print_stats()
    result=result.getvalue()
    result='ncalls'+result.split('ncalls')[-1]
    result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
    now = datetime.now()
    filename = now.strftime("%m-%d-%Y-%H-%M-%S") + "_" + filename  + '.csv'
    create_directory("data/timing/")
    filename = "data/timing/" + filename.replace("'","")
    with open(filename, 'w+') as f:
        f.write(result)
        f.close()
    df = pd.read_csv(filename)
    print(df.columns)
    df = df.sort_values(by=["cumtime"], ascending=False)
    df["filename:lineno(function)"] = [re.sub("^.*[\\\\]","",word)[50:] for word in df["filename:lineno(function)"]]
    df.to_csv(filename)
    #df = df[:10]
    #df["filename:lineno(function)"] = [word.split(":")[2] for word in df["filename:lineno(function)"] if len(word.split(":")) >= 2]
    print(df[:30])
    

def show_profile():
    
    global filename
    print(filename)
    
    

if __name__ == "__main__":

    if PROFILER==True:
        profile_run()
        show_profile()
    else:
        main()
