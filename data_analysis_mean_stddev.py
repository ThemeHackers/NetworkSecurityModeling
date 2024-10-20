import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore , Style
import random
import pyfiglet
import colorama

colorama.init(autoreset=True)

def rainbow_text(text):
    colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]
    rainbow_str = ""
    for char in text:
        rainbow_str += random.choice(colors) + char
    return rainbow_str + Style.RESET_ALL
ascii_art = pyfiglet.figlet_format("DATA ANALYSIS", width=100)
print(rainbow_text(ascii_art))

data = [97.10, 97.30, 97.10, 97.52, 96.98, 97.32, 97.18, 97.16, 96.93, 97.06] # This information is obtained from ten self-training sessions.
mean_value = np.mean(data)
std_dev_value = np.std(data)
x_values = range(1, len(data) + 1)
plt.figure(figsize=(10,5))
plt.plot(x_values, data, label='Data', marker='o', linestyle='-')
plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.2f}%')
plt.axhline(y=mean_value + std_dev_value, color='g', linestyle='--', label=f'Mean + Std Dev = {mean_value + std_dev_value:.2f}%')
plt.axhline(y=mean_value - std_dev_value, color='g', linestyle='--', label=f'Mean - Std Dev = {mean_value - std_dev_value:.2f}%')
plt.title("Mean and Standard Deviation Plot / 100%")
plt.xlabel("Data Point Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig('data_analysis_mean_stddev.png')
print(Fore.GREEN + 'Graph created successfully')
plt.show()


