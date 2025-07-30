# main.py
import cv2
import numpy as np
import time
import yaml
import tkinter as tk
from tkinter import messagebox
import threading

from system_initializer import MeasurementSystem

system = MeasurementSystem()
system.run()