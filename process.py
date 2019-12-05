import subprocess
COMMAND = "python display.py -r results/forecast/week{}.p -o results/forecast/simulated/week{}.png predict -b4000 -t1"
for i in range(2, 33, 2):
    print(COMMAND.format(i,i))
    subprocess.run(COMMAND.format(i, i), shell=True)
