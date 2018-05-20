.. currentmodule:: pyplanscoring.constraints

DVH metrics
===============

This section provides extended aplications of :obj:`pyplanscoring` API.
PyPlanscoring has methods to extract dose-volume histogram metrics with regular expression operators
It improves the ability to use computer algorithms to automate calculation of DVH metrics.
This functinality is provided by the package :obj:`pyplanscoring.constraints`.

It is possible to handle a single structure DVH using the class :obj:`DVHMetrics`

The recommended nomenclature is described on section 9 of the `AAPM Report No. 263 <https://www.aapm.org/pubs/reports/RPT_263.pdf>`_.

The user must set metrics in string format. Example:

.. code-block:: python

    from pyplanscoring import PyPlanScoringAPI, DVHMetrics

    # DVH calculation use-case
    # RS file
    rs_file = 'RT-Structure.dcm'
    # RD file
    rd_file = 'RT-DOSE.dcm'

    pp = PyPlanScoringAPI(rs_file, rd_file)

    #calculation parameters
    calc_grid = (0.2, 0.2, 0.2)  # mm3

    # calculating one structure DVH using roi_number
    dvh = pp.get_structure_dvh(roi_number=2, calc_grid=calc_grid)

    # getting DVH metrics
    dvh_metrics = DVHMetrics(dvh)
    metrics = ['Min[Gy]',
               'Mean[Gy]',
               'Max[Gy]',
               'D99%[Gy]',
               'D95%[Gy]',
               'D5%[Gy]',
               'D1%[Gy]',
               'D0.03cc[Gy]',
               'V25.946Gy[cc]']

    results = [dvh_metrics.execute_query(metric) for metric in metrics]

    print(results)

That's it! You can move on to the :doc:`user manual <../user_manual>` to see what
part of this library interests you.
