import os 

os.environ["DICP_TOPS_DEVICE_ID"] = "3"
os.environ["DICP_TOPS_DIPU"] = "False"

dir_name = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_name)

files = os.listdir(dir_name)

files.remove("resnet_performance.py")
files.remove("resnet_precision.py")
files.remove("test_all.py")

len = 0
files.sort()

for file_name in files[27:]:
    if "dipu" in file_name or "bwd" in file_name:
        continue
    len += 1
    print(f"TEST: {file_name} running... ", flush=True)
    res = os.system(f"python {file_name}")
    if res != 0:
        print(f"Run {file_name} failed.", flush=True)
        break
    print(f"TEST: {file_name} passed.", flush=True)
    
print(f"All {len} tests completed.")
    