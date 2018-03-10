import sys
import os

from pyplanscoring.complexity.PyComplexityMetric import PyComplexityMetric
from pyplanscoring.complexity.dicomrt import RTPlan
import matplotlib.pyplot as plt# if len(sys.argv) != 2:
#     print("Usage: %s path to DICOM RT-PLAN file *.dcm" % (sys.argv[0]))
#     sys.exit(1)

pfile = r"/home/victor/Dropbox/Plan_Competition_Project/competition_2018/Lung Files - Send to Victor - March 1st 2018/87.9 Score - With Plan IQ/RP.1.2.246.352.71.5.584747638204.1034529.20180301221910.dcm"

plan_info = RTPlan(filename=pfile)
plan_dict = plan_info.get_plan()
beams = [beam for k, beam in plan_dict['beams'].items()]
complexity_obj = PyComplexityMetric()

complexity_metric = complexity_obj.CalculateForPlan(None, plan_dict)
complexity_per_beam = complexity_obj.CalculateForPlanPerBeam(None, plan_dict)
complexity_per_beam_cp = [complexity_obj.CalculateForBeamPerAperture(None, plan_dict, bi) for bi in beams]
apertures_per_beam = [complexity_obj.CreateApertures(None, plan_dict, bi) for bi in beams]

_, plan_file = os.path.split(pfile)

beam0_ang =[a.GantryAngle for a in apertures_per_beam[0]]
plt.plot(complexity_per_beam_cp[0])
plt.xlabel('Control Point')
plt.ylabel('CI [mm-1]')
plt.title("Beam 1")
plt.show()


print("Reference: https://github.com/umro/Complexity")
print("Python version by Victor Gabriel Leandro Alves, D.Sc. - victorgabr@gmail.com")
print("Plan %s aperture complexity: %1.3f [mm-1]: " % (plan_file, complexity_metric))
