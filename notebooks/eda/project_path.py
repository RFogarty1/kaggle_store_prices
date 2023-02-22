

#https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im


import sys
import os

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir, "shared_code"))
if module_path not in sys.path:
    sys.path.append(module_path)

secondPath = os.path.abspath(os.path.join(os.pardir, os.pardir, "shared_code", "pipeline"))
if secondPath not in sys.path:
	sys.path.append(secondPath)

