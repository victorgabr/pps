from collections import OrderedDict


def test_1(analytical_data, test1_data, lim=3, save_data=False):
    """
    In Test 1, the axial contour spacing was kept constant at
    0.2 mm to essentially eliminate the variation and/or errors
    associated with rendering axial contours into volumes, and to
    focus solely on the effect of altering the dose grid resolution
    in various stages from fine (0.4 × 0.2 × 0.4 mm 3 ) to coarse (3
    × 3 × 3 mm 3 ).
    Analytical results for the following parameters
    per structure were compared to both PlanIQ (with supersam-
    pling turned on: Ref. 20) and PINCACLE: total volume (V );
    mean, maximum, and minimum dose (D mean , D max , D min );
    near-maximum (D1, dose covering 1% of the volume) and
    near-minimum (D99) doses; D95 and D5; and maximum dose
    to a small absolute (0.03 cm 3 ) volume (D0.03 cm 3 ). We were
    primarily interested in the high and low dose regions because
    with the linear dose gradient, they correspond to the structure
    boundary and this is where the deviations are expected to occur.

    Results of Test 1. Dose grid resolution is varied while axial contour
    spacing is kept at 0.2 mm. Numbers of points (n) exceeding 3% difference
    (∆) from analytical are presented along with the range of % ∆. Total number
    of structure/dose combinations is N = 40 (20 for V ).

    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  2.2min finished

    voxel (0.1, 0.1, 0.1) mm

                      count        range
    Total volume (cc)     0  [-0.7, 0.5]
    Dmin                  0  [-0.1, 2.6]
    Dmax                  0  [-0.4, 0.0]
    Dmean                 0  [-0.2, 0.3]
    D99                   0  [-1.9, 1.9]
    D95                   0  [-1.3, 0.4]
    D5                    0  [-0.3, 0.2]
    D1                    0  [-0.1, 0.2]
    D0.03cc               8  [-0.1, 5.8]

    #TODO Check interpotion at small D(0.03) cc

    """

    # DICOM FILES
    structure_dict, dose_files_dict = test1_data

    # grab analytical data
    mask = analytical_data['CT slice spacing (mm)'] == '0.2mm'
    df = analytical_data.loc[mask]

    # Constrains to get data
    constrains = OrderedDict()
    constrains['Total_Volume'] = True
    constrains['min'] = 'min'
    constrains['max'] = 'max'
    constrains['mean'] = 'mean'
    constrains['D99'] = 99
    constrains['D95'] = 95
    constrains['D5'] = 5
    constrains['D1'] = 1
    constrains['Dcc'] = 0.03

    # GET CALCULATED DATA
    # backend = 'threading'
    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(calc_data)(row,
                           dose_files_dict,
                           structure_dict,
                           constrains,
                           calculation_options) for row in df.iterrows())

    # aggregating data
    df_concat = [d[0] for d in res]
    sname = [d[1] for d in res]

    result = pd.concat(df_concat, axis=1).T.reset_index()
    result['Structure name'] = sname

    res_col = ['Structure name', 'Dose Voxel (mm)', 'Gradient direction', 'Total volume (cc)', 'Dmin', 'Dmax', 'Dmean',
               'D99', 'D95', 'D5', 'D1', 'D0.03cc']
    num_col = ['Total volume (cc)', 'Dmin', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    df_num = df[num_col]
    result_num = result[result.columns[1:-2]]
    result_num.columns = df_num.columns
    delta = ((result_num - df_num) / df_num) * 100

    # print table

    res = OrderedDict()
    for col in delta:
        count = np.sum(np.abs(delta[col]) > lim)
        rg = np.array([round(delta[col].min(), 1), round(delta[col].max(), 1)])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T
    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/core/validation_paper'
    if save_data:
        result.to_excel(os.path.join(dest, 'Test_1_result.xls'))
        test_table.to_excel(os.path.join(dest, 'test_1_table_paper.xls'))

    print(test_table)
    return test_table


def test2(lim=3):
    """
                              count         range
        Total volume (cc)     2   [-3.9, 0.6]
        Dmin                  0   [-0.2, 2.6]
        Dmax                  0   [-0.4, 0.0]
        Dmean                 0   [-0.8, 0.7]
        D99                   8  [-14.4, 5.2]
        D95                   2   [-4.2, 3.2]
        D5                    0   [-0.7, 0.9]
        D1                    0   [-1.1, 2.7]
        D0.03cc              11   [0.2, 10.0]


    """
    ref_data = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_data.xlsx'
    struc_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/STRUCTURES'
    dose_grid_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS'

    snames = ['Sphere_10_0', 'Sphere_20_0', 'Sphere_30_0',
              'Cylinder_10_0', 'Cylinder_20_0', 'Cylinder_30_0',
              'RtCylinder_10_0', 'RtCylinder_20_0', 'RtCylinder_30_0',
              'Cone_10_0', 'Cone_20_0', 'Cone_30_0',
              'RtCone_10_0', 'RtCone_20_0', 'RtCone_30_0']

    structure_path = [os.path.join(struc_dir, f + '.dcm') for f in snames]

    structure_dict = dict(zip(snames, structure_path))

    dose_files = [os.path.join(dose_grid_dir, f) for f in [
        'Linear_AntPost_1mm_Aligned.dcm',
        'Linear_AntPost_2mm_Aligned.dcm',
        'Linear_AntPost_3mm_Aligned.dcm',
        'Linear_SupInf_1mm_Aligned.dcm',
        'Linear_SupInf_2mm_Aligned.dcm',
        'Linear_SupInf_3mm_Aligned.dcm']]

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'1': dose_files[0], '2': dose_files[1], '3': dose_files[2]},
        'Y(SI)': {'1': dose_files[3], '2': dose_files[4], '3': dose_files[5]}}

    # grab analytical data

    df = pd.read_excel(ref_data, sheetname='Analytical')

    dfi = df.ix[40:]
    mask0 = dfi['Structure Shift'] == 0
    dfi = dfi.loc[mask0]

    # Constrains to get data
    # Constrains

    constrains = OrderedDict()
    constrains['Total_Volume'] = True
    constrains['min'] = 'min'
    constrains['max'] = 'max'
    constrains['mean'] = 'mean'
    constrains['D99'] = 99
    constrains['D95'] = 95
    constrains['D5'] = 5
    constrains['D1'] = 1
    constrains['Dcc'] = 0.03

    # GET CALCULATED DATA
    res = Parallel(n_jobs=-1, verbose=11)(
        delayed(calc_data)(row,
                           dose_files_dict,
                           structure_dict,
                           constrains,
                           calculation_options) for row in dfi.iterrows())

    # aggregating data
    df_concat = [d[0] for d in res]
    sname = [d[1] for d in res]

    result = pd.concat(df_concat, axis=1).T.reset_index()
    result['Structure name'] = sname

    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/core/validation_paper'
    result.to_excel(os.path.join(dest, 'Test_2_result.xls'))

    res_col = ['Structure name', 'Dose Voxel (mm)', 'Gradient direction', 'Total volume (cc)', 'Dmin', 'Dmax', 'Dmean',
               'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    num_col = ['Total volume (cc)', 'Dmin', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1', 'D0.03cc']

    df_num = dfi[num_col]

    result_num = result[result.columns[1:-2]]
    result_num.columns = df_num.columns
    result_num.index = df_num.index

    delta = ((result_num - df_num) / df_num) * 100

    pcol = ['Total volume (cc)', 'Dmax', 'Dmean', 'D99', 'D95', 'D5', 'D1']

    res = OrderedDict()

    for col in delta:
        count = np.sum(np.abs(delta[col]) > lim)
        rg = np.array([round(delta[col].min(), 1), round(delta[col].max(), 1)])
        res[col] = {'count': count, 'range': rg}

    test_table = pd.DataFrame(res).T

    test_table.to_excel(os.path.join(dest, 'test_2_table_paper.xls'))

    print(test_table)


def test3(plot_curves=True):
    """
                       Gradient Resolution (mm)  max  mean  min  std
    Sphere_10_0        Z(AP)               1 -0.0  -0.2 -0.3  0.1
    Sphere_10_0        Y(SI)               1  0.0  -0.2 -0.4  0.2
    Sphere_30_0        Z(AP)               3 -0.0  -1.1 -1.9  0.6
    Sphere_30_0        Y(SI)               3  0.3  -1.0 -2.1  0.7
    Cylinder_10_0      Z(AP)               1  0.5   0.3 -0.1  0.2
    Cylinder_10_0      Y(SI)               1  0.3   0.2 -0.0  0.1
    Cylinder_30_0      Z(AP)               3  0.3   0.1  0.0  0.1
    Cylinder_30_0      Y(SI)               3  0.3   0.2 -0.0  0.1
    RtCylinder_10_0    Z(AP)               1  0.5   0.3  0.0  0.1
    RtCylinder_10_0    Y(SI)               1  0.3   0.0 -0.2  0.1
    RtCylinder_30_0    Z(AP)               3  0.7   0.5  0.3  0.1
    RtCylinder_30_0    Y(SI)               3  1.1   0.3 -0.5  0.4
    Cone_10_0          Z(AP)               1  0.3   0.1 -0.1  0.2
    Cone_10_0          Y(SI)               1  1.0   0.3  0.2  0.2
    Cone_30_0          Z(AP)               3 -0.0  -0.8 -1.4  0.4
    Cone_30_0          Y(SI)               3  0.9  -1.3 -1.5  0.5
    RtCone_10_0        Z(AP)               1  0.0  -1.1 -1.4  0.3
    RtCone_10_0        Y(SI)               1  0.0  -1.0 -1.4  0.5
    RtCone_30_0        Z(AP)               3 -0.1  -2.5 -3.9  1.3
    RtCone_30_0        Y(SI)               3  0.5  -2.5 -4.4  1.8
    Average (N = 5)    Y(SI)               1  0.3  -0.1 -0.4  0.2
    Average (N = 5)    Z(AP)               1  0.3  -0.1 -0.4  0.2
    Average (N = 5)    Y(SI)               3  0.6  -0.9 -1.7  0.7
    Average (N = 5)    Z(AP)               3  0.2  -0.8 -1.4  0.5

    """

    ref_data = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/analytical_data.xlsx'

    struc_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/STRUCTURES'
    dose_grid_dir = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/DVH-Analysis-Data-Etc/DOSE GRIDS'
    st = 2

    snames = ['Sphere_10_0', 'Sphere_30_0',
              'Cylinder_10_0', 'Cylinder_30_0',
              'RtCylinder_10_0', 'RtCylinder_30_0',
              'Cone_10_0', 'Cone_30_0',
              'RtCone_10_0', 'RtCone_30_0']

    structure_path = [os.path.join(struc_dir, f + '.dcm') for f in snames]

    structure_dict = dict(zip(snames, structure_path))

    dose_files = [os.path.join(dose_grid_dir, f) for f in [
        'Linear_AntPost_1mm_Aligned.dcm',
        'Linear_AntPost_2mm_Aligned.dcm',
        'Linear_AntPost_3mm_Aligned.dcm',
        'Linear_SupInf_1mm_Aligned.dcm',
        'Linear_SupInf_2mm_Aligned.dcm',
        'Linear_SupInf_3mm_Aligned.dcm']]

    # dose dict
    dose_files_dict = {
        'Z(AP)': {'1': dose_files[0], '2': dose_files[1], '3': dose_files[2]},
        'Y(SI)': {'1': dose_files[3], '2': dose_files[4], '3': dose_files[5]}}

    test_files = {}
    for s_name in structure_dict:
        grad_files = {}
        for grad in dose_files_dict:
            tick = str(int(int(re.findall(r'\d+', s_name)[0]) / 10))
            grad_files[grad] = dose_files_dict[grad][tick]

        test_files[s_name] = grad_files

    result = OrderedDict()
    for sname in snames:
        struc_path = structure_dict[sname]
        # set structure's object
        struc = ScoringDicomParser(filename=struc_path)
        structures = struc.GetStructures()
        structure = structures[st]

        # set end cap by 1/2 slice thickness
        calculation_options['end_cap'] = structure['thickness'] / 2.0

        # set up sampled structure
        struc_teste = Structure(structure, calculation_options)
        str_result = {}
        test_data = test_files[sname]
        for k in test_data:
            # get dose
            dose_file = test_data[k]
            dicom_dose = ScoringDicomParser(filename=dose_file)
            dhist, chist = struc_teste.calculate_dvh(dicom_dose)
            dvh_data = struc_teste.get_dvh_data()
            str_result[k] = dvh_data

        result[sname] = str_result

    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/testdata/test3_ref_dvh.obj'
    # save(an_data, dest)
    an_data = load(dest)

    teste = []
    curve_compare = []
    for s in result:
        for g in result[s]:
            adata = an_data[s][g]
            calc_data = result[s][g]
            cmp = CurveCompare(adata['dose_axis'], adata['data'], calc_data['dose_axis'], calc_data['data'])
            curve_stats = cmp.stats_paper
            curve_stats['Resolution (mm)'] = str(int(int(re.findall(r'\d+', s)[0]) / 10))
            curve_stats['Gradient'] = g
            curve_compare.append(cmp)
            tmp = pd.DataFrame(curve_stats, index=[s])
            teste.append(tmp)

    df_final = pd.concat(teste)

    mask0 = np.logical_and(df_final['Resolution (mm)'] == '1', df_final['Gradient'] == 'Y(SI)')
    mask1 = np.logical_and(df_final['Resolution (mm)'] == '1', df_final['Gradient'] == 'Z(AP)')
    mask2 = np.logical_and(df_final['Resolution (mm)'] == '3', df_final['Gradient'] == 'Y(SI)')
    mask3 = np.logical_and(df_final['Resolution (mm)'] == '3', df_final['Gradient'] == 'Z(AP)')

    # Row 0
    r0 = pd.DataFrame(['Y(SI)'], index=['Average (N = 5)'], columns=['Gradient'])
    r0['Resolution (mm)'] = '1'
    ri = pd.DataFrame(df_final[mask0].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r0 = r0.join(ri)

    # Row 1
    r1 = pd.DataFrame(['Z(AP)'], index=['Average (N = 5)'], columns=['Gradient'])
    r1['Resolution (mm)'] = '1'
    ri = pd.DataFrame(df_final[mask1].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r1 = r1.join(ri)

    # Row 2
    r2 = pd.DataFrame(['Y(SI)'], index=['Average (N = 5)'], columns=['Gradient'])
    r2['Resolution (mm)'] = '3'
    ri = pd.DataFrame(df_final[mask2].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r2 = r2.join(ri)

    # Row 3
    r3 = pd.DataFrame(['Z(AP)'], index=['Average (N = 5)'], columns=['Gradient'])
    r3['Resolution (mm)'] = '3'
    ri = pd.DataFrame(df_final[mask3].mean().round(1)).T
    ri.index = ['Average (N = 5)']
    r3 = r3.join(ri)
    result_df = pd.concat([df_final, r0, r1, r2, r3])

    print(result_df)
    dest = '/home/victor/Dropbox/Plan_Competition_Project/pyplanscoring/core/validation_paper'
    result_df.to_excel(os.path.join(dest, 'test_3_table.xls'))
    #
    # result_df.to_excel('test_3_table.xls')

    if plot_curves:
        # for c in curve_compare:
        #     c.plot_results()
        for grad in ['Z(AP)', 'Y(SI)']:
            for s_key in result:
                adata = an_data[s_key][grad]
                calc_data = result[s_key][grad]
                fig, ax = plt.subplots()
                ax.plot(adata['dose_axis'], adata['data'], label='Analytical')
                ax.plot(calc_data['dose_axis'], calc_data['data'], label='Software')
                ax.legend(loc='best')
                ax.set_xlabel('Dose (cGy)')
                ax.set_ylabel('volume (cc)')
                title = s_key + ' Dose Gradient ' + grad + '.png'
                ax.set_title(title)
                fig.savefig(os.path.join(dest, title), format='png', dpi=100)
        plt.show()

    return curve_compare, result_df
