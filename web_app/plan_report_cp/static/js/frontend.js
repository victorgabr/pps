$(function () {
    'use strict';

    var report = {
        // rs_filename: null,
        // ct_filename: null,
        rd_filename: null,
        // rd_filename: [],
//        roi_number: null
    };

//    function addStructuresToList(structure_list) {
//        report.structures = structure_list;
//
//        var mandibleSelect = $("#mandible_select");
//        structure_list.forEach(function (structure) {
//            mandibleSelect.append(new Option(structure.name, structure.value));
//        });
//
//        // Try to see if mandible exists in list of structures
//        var mandible_id = structure_list.map(function (el) {return el.name}).indexOf("Mandible");
//        mandibleSelect.val(mandible_id);
//    }

    function validateReport() {
//        report.roi_number = $("#mandible_select").val();
        if (report.rd_filename) {
            $("#create_report").removeAttr("disabled");
        } else {
            $("#create_report").attr("disabled", "disabled");
        }
    }

//     $('#structure_upload').fileupload({
//         url: '/api/v1/get_structure_names/',
//         add: function(e, data) {
//             $("#rs_progress").removeClass("hide");
//             $("#structure_upload").attr("disabled", "disabled");
//             $("#structure_upload_button").addClass("disabled");
//             data.submit();
//         },
//         done: function(e, response){
//             var current_data = JSON.parse(response.result);
//             report.rs_filename = current_data.rs_filename;
// //            addStructuresToList(current_data.structure_list);
//             $("#other_upload_container").removeClass("hide");
//             // $("#struct_upload_container").addClass("hide");
//             $("#warning_box").empty();
//         },
//         fail: function(e, response) {
//             console.log(response.result);
//             $("#structure_upload").removeAttr("disabled");
//             $("#structure_upload_button").removeClass("disabled");
//             $("#rs_progress .progress-bar").css('width', 0 + "%");
//             $("#rs_progress").addClass("hide");
//             $("#warning_box").append('<div class="alert alert-danger alert-dismissable dose-alert" role="alert">The uploaded file is not an RTStruct file.</div>');
//         },
//         progressall: function (e, data) {
//             var progress = parseInt(data.loaded / data.total * 100, 10);
//             $('#rs_progress .progress-bar').css(
//                 'width',
//                 progress + '%'
//             );
//         }
//     });

//     $('#dose_upload_input').fileupload({
//         url: '/api/v1/upload_dose/',
//         done: function(e, response){
//             var current_data = JSON.parse(response.result);
//             report.rd_filename.push(current_data.rd_filename);
//             console.log(report.rd_filename);
//             validateReport()
//
//             $("#dose_progress").addClass("hide");
//             $("#dose_upload_button").addClass("hide");
//             // $("#dose_upload_button").removeClass("disabled");
// //            $("#dose_upload_button").find("span").html("Combine more doses")
//             $("#dose_done").removeClass("hide");
//         },
//         add: function(e, data) {
//             data.files.forEach(function(file){
//                 $("#dose_uploaded_filenames").append('<li>' + file.name + ' <span class="glyphicon glyphicon-remove reset-dose" aria-hidden="true"></span></li>')
//             })
//             data.submit();
//             if (data.files[0].name.indexOf("RD") == -1) {
//                 $("#warning_box").append('<div class="alert alert-warning alert-dismissable dose-alert" role="alert">Warning: Dose filename does not contain "RD" which may indicate a non-dose file.</div>');
//             }
//             $("#dose_progress").removeClass("hide");
//             $("#dose_upload_button").addClass("disabled");
//         },
//         fail: function(e, response) {
//             console.log(response.result);
//             $("#dose_upload_button").removeClass("disabled");
//         },
//         progressall: function (e, data) {
//             var progress = parseInt(data.loaded / data.total * 100, 10);
//             $('#dose_progress .progress-bar').css(
//                 'width',
//                 progress + '%'
//             );
//         },
//     });


    $('#ct_upload_input').fileupload({
        url: '/upload_dose',
        done: function(e, response){
            var current_data = JSON.parse(response.result);
            report.rd_filename = current_data.rd_filename;
            validateReport();

            $("#ct_progress").addClass("hide");
            $("#ct_upload_button").addClass("hide");
            $("#ct_done").removeClass("hide");
            $("#ct_upload_input").removeAttr("disabled");
            $("#ct_upload_button").removeClass("disabled");
        },
        fail: function(e, response) {
            console.log(response.result);
            $("#ct_upload_button").removeClass("disabled");
        },
        add: function(e, data) {
            data.files.forEach(function(file){
                $("#ct_uploaded_filenames").html("<li>" + file.name + "</li>")
            })
            data.submit();
            if (data.files[0].name.indexOf("RD") == -1) {
                $("#warning_box").append('<div class="alert alert-warning alert-dismissable ct-alert" role="alert">Warning: Filename does not contain "RD" which may indicate a non-DICOM-RT Dose file.</div>');
            }
            $("#ct_progress").removeClass("hide");
            $("#ct_upload_button").addClass("disabled");
        },
        progressall: function (e, data) {
            var progress = parseInt(data.loaded / data.total * 100, 10);
            $('#ct_progress .progress-bar').css(
                'width',
                progress + '%'
            );
        },
    });

    // $("body").off("click", ".reset-dose").on("click", ".reset-dose", function (e) {
    //     var dose_filename = $.trim($(e.currentTarget).parent()[0].innerText);
    //     $(e.currentTarget).parent().remove();
    //     $("#dose_done").addClass("hide");
    //     $("#warning_box").find(".dose-alert").remove();
    //
    //     var index = report.rd_filename.indexOf(dose_filename);
    //     report.rd_filename.splice(index, 1);
    //
    //     if (report.rd_filename.length == 0) {
    //         $("#dose_upload_button").find("span").html("Select DICOM Dose file")
    //     }
    //     validateReport();
    //     return false;
    // });

    $("#reset_ct").click(function (e) {
        $("#ct_upload_button").removeClass("hide");
        $("#ct_uploaded_filenames").empty();
        $("#ct_done").addClass("hide");
        $("#warning_box").find(".ct-alert").remove();

        report.ct_filename = null;
        validateReport();
    });

    $("#create_report").click(function (e){
        e.preventDefault();
        // var roi_number = $("#mandible_select").val();
        if (report.rd_filename) {
            // $("#rd_input").val(JSON.stringify(report.rd_filename));
            $("#rd_input").val(report.rd_filename);
            // $("#ct_input").val(report.ct_filename);
            // $("#rs_input").val(report.rs_filename);
            // $("#roi_input").val(roi_number);
            $("#create_report").attr("disabled", "disabled");
            document.getElementById("reportform").submit();
        }
    });
});
