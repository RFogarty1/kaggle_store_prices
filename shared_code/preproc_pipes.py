
import os
import math

import numpy as np
import pandas as pd

import sklearn as sk
import sklearn.pipeline


_THIS_DIR = os.path.split( os.path.abspath(__file__) )[0]
_RAW_DATA_DIR = os.path.join( _THIS_DIR, "..", "raw_data" )


#
class AddLog1pSales():

	def __init__(self):
		pass

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		outX["sales_log1p"] = np.log1p(outX["sales"])
		return outX

#Add a lagged moving average column
class AddLeftMovingAverage():
	
	def __init__(self, windows, lags, targCol="sales"):
		self.windows = windows
		self.lags = lags
		self.targCol = targCol
		
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		outX = outX.sort_values(["store_nbr","family","date"])
		for window in self.windows:
			for lag in self.lags:
				outCol = self.targCol + "_l{}".format(int(lag)) + "_ma{}".format(int(window))
				outX[outCol] = outX.groupby(["store_nbr","family"]).shift(lag)[self.targCol]
				outX[outCol] = outX[outCol].rolling(window,min_periods=1).mean()
		return outX




#
class AddFractProm():

	def __init__(self):
		pass

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		if not "store_promotions" in outX.columns:
			outX = AddStoreWideSums(promSum=True,salesSum=False).transform(outX)

		outX["fract_promotions"] = 0.0		
		def _applyFunct(inpRow):
			if inpRow.store_promotions == 0:
				inpRow.fract_promotions = 0
			else:
				inpRow.fract_promotions = inpRow.onpromotion / inpRow.store_promotions
			return inpRow

		outX = outX.apply(_applyFunct, axis=1)
		return outX

#
class AddNumbTransactionsData():
	
	def __init__(self):
		pass
	
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		#Get transactions data
		transactionsFile = os.path.join(_RAW_DATA_DIR,"transactions.csv")
		transFrame = pd.read_csv(transactionsFile)
		transFrame["date"] = pd.to_datetime(transFrame["date"])
	
		#
		useX = inpX.copy()
		useX["date"] = pd.to_datetime(useX["date"])
		_currKwargs = {"left_on":["date","store_nbr"], "right_on":["date","store_nbr"],"how":"left"}
		outFrame = pd.merge(useX, transFrame, **_currKwargs)
		
		#Impute any missing values
		if not "store_sales" in outFrame.columns:
			outFrame = AddStoreWideSums(promSum=False).transform(outFrame) 
		meanDict = transFrame.groupby(["store_nbr"]).mean()["transactions"].to_dict()
		
		def applyFunct(inpRow):
			transVal = inpRow.transactions
			storeSales = inpRow.store_sales
			if pd.isna(transVal):
				if storeSales < 1:
					inpRow.transactions = 0
				else:
					inpRow.transactions = meanDict[inpRow.store_nbr]
			return inpRow

		outFrame.update( outFrame.loc[ outFrame["transactions"].isna() ].apply(applyFunct,axis=1) )
		return outFrame


#Add some store wide sums
class AddStoreWideSums():
	
	def __init__(self, promSum=True, salesSum=True):
		self.promSum = promSum
		self.salesSum = salesSum
		
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		useX = inpX.copy()
		
		#Compute
		useX["date"] = pd.to_datetime(useX["date"])
		summedFrame = useX.groupby(["date","store_nbr"]).sum().reset_index()
		summedFrame = summedFrame.rename( self._getRenameCols() ,axis=1)
		
		#Join
		joinCols = ["date", "store_nbr"]
		_currKwargs = {"left_on":joinCols,"right_on":joinCols, "how":"left", "suffixes":(None,"_y")}
		outCols = ["date","store_nbr"] + [val for val in self._getRenameCols().values()]
		outFrame = pd.merge(useX, summedFrame[outCols], **_currKwargs)
		
	
		return outFrame
	
	def _getRenameCols(self):
		outDict = dict()
		if self.promSum:
			outDict["onpromotion"] = "store_promotions"
		if self.salesSum:
			outDict["sales"] = "store_sales"
		return outDict


#Join various store info data
class AddStoreInfoData():

	def __init__(self):
		pass

	def fit(self, inpX, inpY=None):
		return self

	def transform(self, inpX):
		#
		storeInfoFile = os.path.join(_RAW_DATA_DIR, "stores.csv")
		storeInfoData = pd.read_csv(storeInfoFile)

		_renameDict = {"city":"store_city", "state":"store_state", "type":"store_type", "cluster":"store_cluster"}
		storeInfoData = storeInfoData.rename(_renameDict,axis=1)

		#Handle the join and return
		outX = inpX.copy()
		_currKwargs = {"left_on":"store_nbr", "right_on":"store_nbr", "how":"left"}
		outX = pd.merge(outX,storeInfoData, **_currKwargs)
		return outX

#Join oil prices data
class AddOilPriceData():
	
	def __init__(self, window=10):
		self.window = window
	
	def fit(self, inpX, inpY=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		outX["date"] = pd.to_datetime(outX["date"])
		outX = pd.merge(outX, self._getData(), how="left")
		outX = outX.rename({"rolling":"oil_price_w{}".format(self.window)}, axis=1)

		return outX
	
	def _getData(self):
		oilPath = os.path.join( _RAW_DATA_DIR, "oil.csv" )
		oilData = pd.read_csv(oilPath)		
		oilData.loc[  0, "dcoilwtico" ] = oilData.loc[  1, "dcoilwtico" ]

		#Fill in any missing dates (with knowledge that we KNOW the start and end date are there)
		oilData = oilData.set_index("date")
		oilData.index = pd.DatetimeIndex(oilData.index)
		idxRange = pd.date_range(oilData.index.min(),oilData.index.max())
		oilData = oilData.reindex(idxRange)
		oilData.index.name = "date"

		#A moving average should let all the NaN disapear as long as window is sufficient
		oilData["rolling"] = oilData["dcoilwtico"].rolling(window=self.window, min_periods=0).mean()
		oilData = oilData.reset_index()

		return oilData[["date","rolling"]]

#
class AddLagFeats():
	
	def __init__(self, lagNumbs, targCol="sales"):
		self.lagNumbs = lagNumbs
		self.targCol = targCol
		
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX):
		outX = inpX.copy()
		outX = outX.sort_values(["store_nbr","family","date"])
		for lag in self.lagNumbs:
			outCol = self.targCol + "_l{}".format(lag)
			outX[outCol] = outX.groupby(["store_nbr","family"])[self.targCol].shift(lag)
		return outX

#
class OrdEncodeStoreType():

	def __init__(self):
		pass

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
		mapDict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
		outX["store_type_ordE"] = outX["store_type"].map(lambda x:mapDict[x])
		return outX

#
class StoreStateOrdEncode():

	def __init__(self):
		pass

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
#		states = ['Santa Elena', 'Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
#		       'Santo Domingo de los Tsachilas', 'Bolivar', 'Tungurahua',
#		       'Guayas', 'Los Rios', 'Azuay', 'Loja', 'El Oro', 'Esmeraldas',
#		       'Manabi']
		states = ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
		       'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
		       'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
		       'El Oro', 'Esmeraldas', 'Manabi']
		mapDict = {key:idx for idx,key in enumerate(states)}
		outX["store_state_ordE"] = outX["store_state"].map(lambda x:mapDict[x])
		return outX

#
class StoreCityOrdEncode():

	def __init__(self):
		pass

	def fit(self, inpX, y=None):
		return self

	def transform(self, inpX):
		outX = inpX.copy()
#		city = ['Salinas', 'Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra',
#		       'Santo Domingo', 'Guaranda', 'Ambato', 'Guayaquil', 'Daule',
#		       'Babahoyo', 'Quevedo', 'Playas', 'Cuenca', 'Loja', 'Machala',
#		       'Esmeraldas', 'El Carmen']
		city = ['Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra',
	       'Santo Domingo', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil',
	       'Salinas', 'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad',
	       'Cuenca', 'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen']
		mapDict = {key:idx for idx,key in enumerate(city)}
		outX["store_city_ordE"] = outX["store_city"].map(lambda x:mapDict[x])
		return outX

#store_city 	store_state 	store_type

#Dont need ALL this data; and removing old data is the neatest way to handle
class RemoveDatesNDaysBehindMax():

	def __init__(self, nDays):
		self.nDays = nDays

	def fit(self, inpX, y=None):
		_deltaTime = pd.Timedelta("{}d".format( int(self.nDays) ))
		self.maxDate = pd.to_datetime(inpX["date"]).max() - _deltaTime
		return self

	def transform(self, inpX):
		outFrame = inpX.copy()
		outFrame = outFrame[ pd.to_datetime(outFrame["date"]) >= self.maxDate ] 
		return outFrame

#
class TargEncodeFamilyStore():

	def __init__(self, nDays=None):
		if nDays is not None:
			raise NotImplementedError("")
		self.nDays = nDays

	def fit(self, inpX, y=None):
		useX = inpX.copy()
		_groupedFrame = useX[["family_enc","store_nbr","sales"]].groupby(["family_enc","store_nbr"])
		meanFrame = _groupedFrame.mean().reset_index()
		self.meanDict = meanFrame.set_index(["family_enc","store_nbr"]).to_dict()["sales"]

		#Also have a mean family dict; this is to handle cases where a store hasnt previously sold a family
		self.meanFamDict = useX.groupby("family_enc").mean()["sales"].to_dict()		

		return self

	#TODO: Need a dict encoding for the mean frame maybe; should be A LOT quicker
	def transform(self, inpX):
		outFrame = inpX.copy()
		def _applyFunct(inpRow):
			_family, _store = inpRow.family_enc, inpRow.store_nbr
			try:
				outVal = self.meanDict[ (_family, _store) ]
			except KeyError:
				outVal = self.meanFamDict[ (_family) ]

			return outVal

		outFrame["fam_store_mean_enc"] = outFrame.apply(_applyFunct, axis=1)
		return outFrame


#Want to encode each family with a number; .factorize is annoying so manual mapping it is
class EncodeFamilyArbitrary():
	
	def __init__(self, uniqueFams):
		self._uniqueFams = uniqueFams
		self.encodeDict = self._getEncodeDictFromUniqueFams(uniqueFams)
		self.unknownVal = -1
	
	def _getEncodeDictFromUniqueFams(self, uniqueFams):
		outDict =  {x:idx for idx,x in enumerate( sorted(uniqueFams) )}
		return outDict
		
	def fit(self, inpX, y=None):
		return self
	
	def transform(self, inpX, y=None):
		outX = inpX.copy()
		outX["family_enc"] = inpX["family"].map(lambda x: self.encodeDict.get(x,self.unknownVal) )
		return outX


#
class AddDayOfWeekFeat():
	
	def __init__(self, dateCol="date",featName="day_of_week"):
	    self.dateCol = dateCol
	    self.featName = featName
	    
	def fit(self, inpX, y=None):
	    return self
	
	def transform(self, inpX):
	    outX = inpX.copy()
	    outX[self.featName] = pd.to_datetime(outX[self.dateCol]).map(lambda x:x.day_of_week)
	    return outX

#
class AddDayOfMonth():
	
	def __init__(self, dateCol="date", featName="day_of_month"):
	    self.dateCol = dateCol
	    self.featName = featName
	    
	def fit(self, inpX, y=None):
	    return self
	
	def transform(self, inpX):
	    outX = inpX.copy()
	    outX[self.featName] = pd.to_datetime(outX[self.dateCol]).map(lambda x:x.day)
	    return outX
	
#
class AddDayOfYearSinCos():
	
	def __init__(self, dateCol="date", sinCol="sin_day_of_year", cosCol="cos_day_of_year"):
	    self.dateCol = dateCol
	    self.sinCol = sinCol
	    self.cosCol = cosCol
	
	def fit(self, inpX, y=None):
	    return self
	
	def transform(self, inpX):
	    outX = inpX.copy()
	    def _sinMap(x):
	        return math.sin( 2*math.pi* x.day_of_year/365.25 )
	    def _cosMap(x):
	        return math.cos( 2*math.pi* x.day_of_year/365.25 )
	    
	    outX[self.sinCol] = pd.to_datetime(inpX[self.dateCol]).map( _sinMap )
	    outX[self.cosCol] = pd.to_datetime(inpX[self.dateCol]).map( _cosMap )
	
	    return outX



