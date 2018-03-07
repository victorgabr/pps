def get_calculation_options(ini_file_path):
    """
        Helper method to read app *.ini file
    :param ini_file_path:
    :return:
    """
    import configparser

    # Get calculation defaults
    config = configparser.ConfigParser()
    config.read(ini_file_path)
    calculation_options = dict()
    calculation_options['end_cap'] = config.getfloat('DEFAULT', 'end_cap')
    calculation_options['use_tps_dvh'] = config.getboolean('DEFAULT', 'use_tps_dvh')
    calculation_options['use_tps_structures'] = config.getboolean('DEFAULT', 'use_tps_structures')
    calculation_options['up_sampling'] = config.getboolean('DEFAULT', 'up_sampling')
    calculation_options['maximum_upsampled_volume_cc'] = config.getfloat('DEFAULT', 'maximum_upsampled_volume_cc')
    calculation_options['voxel_size'] = config.getfloat('DEFAULT', 'voxel_size')
    calculation_options['num_cores'] = config.getint('DEFAULT', 'num_cores')
    calculation_options['save_dvh_figure'] = config.getboolean('DEFAULT', 'save_dvh_figure')
    calculation_options['save_dvh_data'] = config.getboolean('DEFAULT', 'save_dvh_data')
    calculation_options['mp_backend'] = config['DEFAULT']['mp_backend']

    return calculation_options
