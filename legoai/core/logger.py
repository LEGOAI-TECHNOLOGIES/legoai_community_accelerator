from datetime import datetime
import os


class Logger:
	logger =None
	log_path = None
	__LOGLEVEL_MAP__ = {
		"CRITICAL" : 50,
		"ERROR" : 40,
		"WARNING" : 30,
		"INFO" : 20,
		"DEBUG" : 10,
		"NOTSET": 0
	}

	@classmethod
	def getLogger(cls,parent_folder_name:str,child_folder_name:str):
		log_path = os.path.join(parent_folder_name,child_folder_name)
		if cls.logger == None or cls.log_path != log_path:
			import logging
			from core.configuration import PATH_CONFIG
			# setting logging
			full_log_path = os.path.join(PATH_CONFIG["CONTAINER_PATH"],log_path)
			filename=os.path.join(PATH_CONFIG["CONTAINER_PATH"],
						 full_log_path,f"{datetime.now().strftime('%d%m%Y')}.log")
			
			logging.basicConfig(
				filename =filename,
				level = cls.__LOGLEVEL_MAP__[PATH_CONFIG["LOG_LEVEL"].upper() ],
				format='%(asctime)s %(message)s',
				filemode='w',
			)
			cls.logger =  logging
			cls.log_path = log_path
			
		return cls.logger
	

