
#  %% calculating DVH
from pyplanscoring import PyPlanScoringAPI, plot_dvh, plot_dvhs, IOHandler
from pyplanscoring import PyDicomParser, PyStructure
from pyplanscoring.vis.contours3d import plot_structure_contours

# DVH calculation use-case
# RS file
rs_file = "/home/victor/Dropbox/Plan_Competition_Project/tests/tests_validation/benchmark_data/DVH-Analysis-Data-Etc/STRUCTURES/Cone_30_0.dcm"
# RD file
rd_file = '/home/victor/Dropbox/Plan_Competition_Project/tests/tests_validation/benchmark_data/DVH-Analysis-Data-Etc/DOSE GRIDS/Linear_AntPost_3mm_Aligned.dcm'

pp = PyPlanScoringAPI(rs_file, rd_file)

# calculation parameters
end_cap_size = 1.5  # mm
calc_grid = (0.2, 0.2, 0.2)  # mm3

# calculating one structure DVH using roi_number
dvh = pp.get_structure_dvh(
    roi_number=2, end_cap=end_cap_size, calc_grid=calc_grid)

# plotting DVH
plot_dvh(dvh, 'My DVH')

# calculating DVH from all strucures in RT-structure file - no oversampling
# dvhs = pp.calc_dvhs(verbose=True)

# Plotting all DVHs in relative volumes
# plot_dvhs(dvhs, 'PyPlanScoring')

# saving results in JSON text
# obj = IOHandler(dvhs)
# output_file_path = 'plan_dvhs.dvh'
# obj.to_json_file(output_file_path)
# %%

rs_file = "/home/victor/Dropbox/Plan_Competition_Project/tests/tests_data/halcyon_plan/RS.1.2.246.352.71.4.818656424269.209262.20180308211033.dcm"
rs_dcm = PyDicomParser(filename=rs_file)

structures = rs_dcm.GetStructures()  # Dict like information of contours
# encapsulate data on PyStructure object
structure = PyStructure(structures[1])

plot_structure_contours(structure.point_cloud, structure.name)
