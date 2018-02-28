import sys
import os

from pyplanscoring.complexity.PyComplexityMetric import PyComplexityMetric
from pyplanscoring.complexity.dicomrt import RTPlan

if len(sys.argv) != 2:
    print("Usage: %s path to DICOM RT-PLAN file *.dcm" % (sys.argv[0]))
    sys.exit(1)

plan_info = RTPlan(filename=sys.argv[1])
plan_dict = plan_info.get_plan()
complexity_metric = PyComplexityMetric().CalculateForPlan(None, plan_dict)
_, plan_file = os.path.split(sys.argv[1])

print("Reference: https://github.com/umro/Complexity")
print("Python version by Victor Gabriel Leandro Alves, D.Sc. - victorgabr@gmail.com")
print("Plan %s aperture complexity: %1.3f [mm-1]: " % (plan_file, complexity_metric))
