import os
import shutil
import copy
import threading

import logging
import logging.handlers

import pydicom 
pydicom.config.convert_wrong_length_to_UN = True

def check_dcm_uniformity(dicom_datasets, modal):
    if not dicom_datasets or len(dicom_datasets) == 1:
        return True

    base_seriesinstanceUID = dicom_datasets[0].SeriesInstanceUID
    base_studyinstanceUID = dicom_datasets[0].StudyInstanceUID

    for ds in dicom_datasets:
        if ds.SeriesInstanceUID != base_seriesinstanceUID or ds.StudyInstanceUID != base_studyinstanceUID or ds.Modality != modal:
            return False

    return True

class Shared_variable:
    def __init__(self, var_type, var_init):
        if not isinstance(var_init, var_type):
            raise ValueError("var_init must have the same type as var_type")
        
        self.init_var = var_init
        self.var = var_init
        self.var_type = var_type
        self.locker = threading.Lock()

    def reset(self):
        with self.locker:
            self.var = self.init_var

    def get(self):
        with self.locker:
            return self.var

    def set(self, value):
        if not isinstance(value, self.var_type):
            raise ValueError("value must have the same type as var_type")
        
        with self.locker:
            self.var = value

class Shared_list:
    def __init__(self, list_init:list=[]):
        self.list = list_init
        self.locker = threading.Lock()
    def clear(self):
        with self.locker:
            self.list.clear()
    def get(self, idx=None):
        with self.locker:
            if idx is None:
                return copy.deepcopy(self.list)
            return self.list[idx]
    def append(self, ds):
        with self.locker:
            self.list.append(ds)

class Thread_list:
    def __init__(self, list_init:list=[]):
        self.list = list_init
        self.locker = threading.Lock()
    def clear(self):
        with self.locker:
            for thd in self.list:
                thd.join()
            self.list.clear()
    def append(self, thd):
        with self.locker:
            if not isinstance(thd, threading.Thread):
                raise ValueError("Appended value for thread list must be a Thread instance")
            self.list.append(thd)
            
class Srv_status:
    def __init__(self, list_init=[]):
        self.analyzing_now = Shared_variable(bool, False)
        self.error = Shared_variable(int, 0)
        self.jobtype = Shared_variable(str, "IDLE")
        self.tracer = Shared_variable(str, "None")
        # self.tracer = Shared_variable(str, "FDG")
        self.report_id = Shared_variable(str, "None")
        self.structural_scan = Shared_variable(str, "None")
        # self.pid = Shared_variable(str, "None")

        self.shared_list = Shared_list(list_init)
        self.threads = Thread_list()

    def reset_all(self):
        self.analyzing_now.reset()
        self.error.reset()
        self.jobtype.reset()
        self.tracer.reset()
        self.report_id.reset()
        self.structural_scan.reset()
        # self.pid.reset()

        self.shared_list.clear()
        self.threads.clear()

    # error
    def set_error(self, value):
        self.error.set(value)

    def get_error(self):
        return self.error.get()

    # analyzing now
    def set_analyzing_now(self, value):
        self.analyzing_now.set(value)

    def isAnalyzing(self):
        return self.analyzing_now.get()

    # shared list
    def reset_list(self):
        self.shared_list.clear()

    def put_to_list(self, ds):
        self.shared_list.append(ds)

    def get_list(self, idx=None):
        return self.shared_list.get(idx)
    
class DailyLoggerWithRotation:
    def __init__(self, args):
        self.log_dir = args.log_dir
        self.log_file = args.log_filename
        self.log_level = logging.INFO
        if args.log_level == "debug":
            self.log_level = logging.DEBUG
        elif args.log_level == "error":
            self.log_level = logging.ERROR
        elif args.log_level == "warn":
            self.log_level = logging.WARNING
        
        self.initialize_logger()

    def initialize_logger(self):
        # Log configuration
        log_formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure TimedRotatingFileHandler (create a new file daily, don't overwrite)
        self.file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=f'{self.log_dir}/{self.log_file}',
            when='midnight',  # Create a new file at midnight daily
            interval=1,  # Daily rotation
            backupCount=30,  # Make backup log files for N days
        )
        self.file_handler.suffix = '%Y-%m-%d'  # Append date to the file name
        self.file_handler.setFormatter(log_formatter)

        # Add the file handler to the root logger
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(self.log_level)
        self.root_logger.addHandler(self.file_handler)

    def detailed_log_format(self, option, desc, event, ret_code):
        if option == 'r':
            return f"> Recv\t\t{event}\t\t{desc}"
        elif option == 'p':
            return f"< Respond\t\t{event}\t{hex(ret_code)}\t{desc}"
        elif option == 's':
            return f"< Send\t\t{event}\t\t{desc}"
        elif option == 'e': # event
            return f"{event}\t\t{desc}"
        elif option == 'c': # nothing
            return f"{desc}\t\t (error code: {ret_code})"
        elif option == 'x': # nothing
            return desc

        return desc

    def info(self, desc, option='x', event=None, ret_code=None):
        logging.info(self.detailed_log_format(option, desc, event, ret_code))

    def error(self, desc, option='x', event=None, ret_code=None):
        logging.error(self.detailed_log_format(option, desc, event, ret_code))

    def warn(self, desc, option='x', event=None, ret_code=None):
        logging.warn(self.detailed_log_format(option, desc, event, ret_code))

    def debug(self, desc, option='x', event=None, ret_code=None):
        logging.debug(self.detailed_log_format(option, desc, event, ret_code))

    def exception(self, exc):
        logging.exception(exc)


def delete_temp_dir(args, app_logger, event_type=None):
    # if args.save_files: # delete temp dir
    if os.path.exists(args.temp_dir):
        for filename in os.listdir(args.temp_dir):
            try:
                file_path = os.path.join(args.temp_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                if event_type is None:
                    app_logger.debug('x', f"Clear the temp directory:\t{file_path}")
                else:
                    app_logger.debug('e', f"Clear the temp directory:\t{file_path}", event_type)
            except Exception as exc:
                if event_type is None:
                    app_logger.error('x', f"Cannot clear the temp directory:\t{file_path}")
                else:
                    app_logger.error('e', f"Cannot clear the temp directory:\t{file_path}", event_type)
                app_logger.exception(exc)


# def add_private_tag(ds, status=None, data_dict=None):
#     from pydicom import dcmread
#     report_filename = '/media/storage4/jl/1025_data_result_ano/PDF/pdf_fbb/test_0726_3.dcm'
#     # report_filename = '/media/storage4/jl/1025_data_result_ano/PDF/pdf_fsf/report_FSF_cbllGM_1.dcm'
#     ds = dcmread(report_filename)

#     block = ds.private_block(0x00bb, "NewcureM", create=True)
#     status = None
#     # if status is not None:

#     # tracer = 'FBB'
#     tracer = 'FDG'
#     out_fileneme = os.path.join(f'/media/storage1/home/user/chu/lab/NCM_BRAIN2.0/analysis_server/temp/{tracer}', os.path.basename(report_filename))
#     if tracer in ['FBB', 'FMM', 'FDG']:

#         # reportID (R_{userID}_{YYYYMMDDHHMMSS}_{5_random_digit})
#         # taskID (T_{userID}_{YYYYMMDDHHMMSS}_{5_random_digit})
#         block.add_new(0x0001, "LO", "1854564_20231211121111_12345")
#         # jobtype ('analysis' / 'denoising')
#         block.add_new(0x0002, "LO", "analysis")
#         # tracer ('FBB', 'FMM', 'FDG', 'FPCIT')
#         block.add_new(0x0003, "SH", f"{tracer}")

#         # polygon series UID
#         block.add_new(0x0004, "UI", )

#         # FileType
#         # ('PT','MR','CT','pdf','roi','fusion','mip','polygon')
#         block.add_new(0x0005, "LO", "pdf")

#         # Analysis method ('Patient_specific_VOI', 'AAL')
#         block.add_new(0x0006, "LO", "Patient_specific_VOI")
#         # Reference region
#         # ('Cerebellum_GM', 'Whole_Cerebellum', 'Whole_Brain', 'Cerebral_WM', 'Pons', 'Occipital_Lobe')
#         block.add_new(0x0007, "LO", "Whole_Cerebellum")

#         # Structural Scan ('CT' or 'MR' or 'None')
#         block.add_new(0x0008, "SH", "MR")

#         # Analyzed PT desc
#         block.add_new(0x0009, "LO", "series desc of original PT dicom")
#         # Analyzed CT or MR desc
#         block.add_new(0x000A, "LO", "series desc of original CT or MR dicom")
#         # Analyzed PT Study Date
#         block.add_new(0x000B, "DA", "19981216")
#         # Analyzed CT or MR Study Date
#         block.add_new(0x000C, "DA", "19981216")

#         ### SUVR (0x0011 ~ 0x001F)
#         # Frontal lobe
#         block.add_new(0x0011, "DS", "1.843")
#         # Parietal lobe
#         block.add_new(0x0012, "DS", "1.843")
#         # Lateral temporal lobe
#         block.add_new(0x0013, "DS", "1.843")
#         # Medial temporal lobe
#         block.add_new(0x0014, "DS", "1.843")
#         # Anterior cingulate gyrus
#         block.add_new(0x0015, "DS", "1.843")
#         # Posterior cingulate gyrus
#         block.add_new(0x0016, "DS", "1.843")
#         # Occipital lobe
#         block.add_new(0x0017, "DS", "1.843")
#         # Caudate
#         block.add_new(0x0018, "DS", "1.843")
#         # Putamen
#         block.add_new(0x0019, "DS", "1.843")
#         # Thalamus
#         block.add_new(0x001A, "DS", "1.843")
#         # Cerebellar gray matter
#         block.add_new(0x001B, "DS", "1.843")
#         # Cerebellar white matter
#         block.add_new(0x001C, "DS", "1.843")
#         # Pons
#         block.add_new(0x001D, "DS", "1.843")
#         # Intensity of reference region
#         block.add_new(0x001E, "DS", "4211.843") 
#         # Composite SUVR
#         block.add_new(0x001F, "DS", "1.843") 
#         ## 요기까지 CSV_tag['SUVR'][레퍼런스 영역] ##
#         if tracer == 'FBB':
#             value = '1.123'
#         elif tracer == 'FDG':
#             value = None

#         ### Centiloid
#         # Frontal lobe (0x0021 ~ 0x002F)
#         block.add_new(0x0021, "DS", value)
#         # Parietal lobe
#         block.add_new(0x0022, "DS", value)
#         # Lateral temporal lobe
#         block.add_new(0x0023, "DS", value)
#         # Medial temporal lobe
#         block.add_new(0x0024, "DS", value)
#         # Anterior cingulate gyrus
#         block.add_new(0x0025, "DS", value)
#         # Posterior cingulate gyrus
#         block.add_new(0x0026, "DS", value)
#         # Occipital lobe
#         block.add_new(0x0027, "DS", value)
#         # Caudate
#         block.add_new(0x0028, "DS", value)
#         # Putamen
#         block.add_new(0x0029, "DS", value)
#         # Thalamus
#         block.add_new(0x002A, "DS", value)
#         # Cerebellar gray matter
#         block.add_new(0x002B, "DS", value)
#         # Cerebellar white matter
#         block.add_new(0x002C, "DS", value)
#         # Pons
#         block.add_new(0x002D, "DS", value)
#         # Intensity of reference region
#         block.add_new(0x002E, "DS", value)
#         # Composite SUVR
#         block.add_new(0x002F, "DS", value)
#         ## 요기까지 CSV_tag['Centiloid'][레퍼런스 영역] ##
#     elif tracer == 'FPCIT':
#         # LVS
#         block.add_new(0x0011, "DS", "1.843")
#         # RVS
#         block.add_new(0x0012, "DS", "1.843")
#         # LAC
#         block.add_new(0x0013, "DS", "1.843")
#         # RAC
#         block.add_new(0x0014, "DS", "1.843")
#         # LAP
#         block.add_new(0x0015, "DS", "1.843")
#         # RAP
#         block.add_new(0x0016, "DS", "1.843")
#         # LPC
#         block.add_new(0x0017, "DS", "1.843")
#         # RPC
#         block.add_new(0x0018, "DS", "1.843")
#         # LPP
#         block.add_new(0x0019, "DS", "1.843")
#         # RPP
#         block.add_new(0x001A, "DS", "1.843")
#         # LVP
#         block.add_new(0x001B, "DS", "1.843")
#         # RVP
#         block.add_new(0x001C, "DS", "1.843")
#         # Intensity of occipital lobe
#         block.add_new(0x001D, "DS", "1.843")

#         # LPP/LAP
#         block.add_new(0x001E, "DS", "1.843")
#         # RPP/RAP
#         block.add_new(0x001F, "DS", "1.843")
#         # LPC/LAC
#         block.add_new(0x0020, "DS", "1.843")
#         # RPC/RAC
#         block.add_new(0x0021, "DS", "1.843")
#         # LPP/LAC
#         block.add_new(0x0022, "DS", "1.843")
#         # RPP/RAC
#         block.add_new(0x0023, "DS", "1.843")
#         # LAC/LVS
#         block.add_new(0x0024, "DS", "1.843")
#         # RAC/RVS
#         block.add_new(0x0025, "DS", "1.843")
#         # LAP/LVS
#         block.add_new(0x0026, "DS", "1.843")
#         # RAP/RVS
#         block.add_new(0x0027, "DS", "1.843")
#         # LAC/LAP
#         block.add_new(0x0028, "DS", "1.843")
#         # RAC/RAP
#         block.add_new(0x0029, "DS", "1.843")
#         # LPP/LVP
#         block.add_new(0x002A, "DS", "1.843")
#         # RPP/RVP
#         block.add_new(0x002B, "DS", "1.843")
#         # LPC/LVP
#         block.add_new(0x002C, "DS", "1.843")
#         # RPC/RVP
#         block.add_new(0x002D, "DS", "1.843")
#         # LPP/LVS
#         block.add_new(0x002E, "DS", "1.843")
#         # RPP/RVS
#         block.add_new(0x002F, "DS", "1.843")
#     print(ds[(0x00bb,0x1021)].value)    
#     ds.save_as(out_fileneme)


# def export_csv():

#     csv_path = '/media/storage1/home/user/chu/lab/NCM_BRAIN2.0/analysis_server/csv_example.csv'
#     cols = ['Frontal lobe','Parietal lobe','Lateral temporal lobe','Medial temporal lobe','Anterior cingulate gyrus','Posterior cingulate gyrus','Occipital lobe','Caudate','Putamen','Thalamus','Cerebellar gray matter','Cerebellar white matter','Pons','intensity of reference region','Composite SUVR']
#     result = []
#     # for postfix in [' (SUVR)', ' (Centiloid)']:
#     #     for c in cols:
#     #         result.append(c+postfix)

#     result = ['LVS','RVS','LAC','RAC','LAP','RAP','LPC','RPC','LPP','RPP','LVP','RVP','Occipital lobe','LPP/LAP','RPP/RAP','LPC/LAC','RPC/RAC','LPP/LAC','RPP/RAC','LAC/LVS','RAC/RVS','LAP/LVS','RAP/RVS','LAC/LAP','RAC/RAP','LPP/LVP','RPP/RVP','LPC/LVP','RPC/RVP','LPP/LVS','RPP/RVS']
#     import csv
#     with open(csv_path, mode='w', newline='') as output_file:
#         csv_writer = csv.writer(output_file)
#         csv_writer.writerow(result)



# if __name__ == "__main__":
    # add_private_tag('a')