import pyodbc

def pullQuery(pull):
	print("[+] connecting to the servers")
	pull = 'set nocount on; ' + pull
	server = '10.75.1.103'#'10.75.6.160'
	cnxn = pyodbc.connect(driver='{SQL Server}', server=server, database='OperationsDataMart', trusted_connection='no',UID = 'evan', PWD = 'evan')
	cursor = cnxn.cursor()
	cursor.execute(pull)
	row = cursor.fetchone()
	results = []
	while row:
		results.append(list(row))
		row = cursor.fetchone()
	cursor.close()
	return results


def pullQueryWithHeader(pull):
	print("[+] connecting to the servers")
	pull = 'set nocount on; ' + pull
	server = '10.75.1.103'#'10.75.6.160'
	cnxn = pyodbc.connect(driver='{SQL Server}', server=server, database='OperationsDataMart', trusted_connection='no',UID = 'evan', PWD = 'evan')
	cursor = cnxn.cursor()
	cursor.execute(pull)
	row = cursor.fetchone()
	results = []
	header = [column[0] for column in cursor.description]
	results.insert(0, header)
	while row:
		results.append(list(row))
		row = cursor.fetchone()
	cursor.close()
	return results

# Only use this if you want to execute something and know what you're doing
#
# def pullQueryExec(pull):
# 	print("[+] connecting to the servers")
# 	server = '10.75.6.160'
# 	cnxn = pyodbc.connect(driver='{SQL Server}', server=server, database='OperationsDataMart', trusted_connection='yes')
# 	cursor = cnxn.cursor()
# 	cursor.execute(pull)
# 	cnxn.commit()
# 	cursor.close()

# def csvToSQL(tableLocation, csvFile, hasHeader=True, delTable=False, calcCol=False):
# 	server = '10.75.6.160'
# 	cnxn = pyodbc.connect(driver='{SQL Server}', server=server, database='TeamOptimizationEngineering',
# 	                      trusted_connection='yes')
# 	print('[+] Connected to ' + server + '\n   Writing to location ' + tableLocation)
# 	with open(csvFile, 'r') as f:
# 		reader = csv.reader(f)
# 		if hasHeader == True:
# 			data = next(reader)
# 		query = 'insert into {}'.format(tableLocation) + ' values ({0})'
# 		cursor = cnxn.cursor()
# 		if delTable == True:
# 			cursor.execute('delete from ' + tableLocation)
# 		if calcCol == True:
# 			import datetime
# 			CalcTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
# 		for data in tqdm(reader):
# 			if calcCol == True:
# 				data.append(CalcTime)
# 			# print(data)
# 			query = query.format(','.join('?' * len(data)))
# 			# print(query)
# 			cursor.execute(query, data)
# 		# try:
# 		#     cursor.execute(query, data)
# 		# except:
# 		#     print(data)
# 		cursor.commit()


if __name__ == '__main__':
	import pandas as pd
	query = '''
		SELECT top 10 [WellName]
		FROM   OperationsDataMart.Dimensions.Wells AS W;
	'''
	raw = pullQueryWithHeader(query)
	header = raw[0]
	data = raw[1:]
	df = pd.DataFrame(data=data, columns=header)
	print(df.head())
