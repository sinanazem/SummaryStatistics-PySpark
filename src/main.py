import pyspark
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql import Window
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName('abc').getOrCreate()



class SummaryStatistics:

    def __init__(self,spark,file_path):

        self.spark = spark
        self.file_path = file_path


        self.data = self.read_data()


    def read_data(self,sep=','):

        try:

            data = self.spark.read.csv(self.file_path,inferSchema=True,header=True,encoding='utf-8',sep=sep)
            return data

        except Exception as e:

            print('reading data failed!')
            print(e)


    @staticmethod
    def TransposeDF(df, columns, pivotCol):

        try:
            columnsValue = list(map(lambda x: str("'") + str(x) + str("',")  + str(x), columns))
            stackCols = ','.join(x for x in columnsValue)
            df_1 = df.selectExpr(pivotCol, "stack(" + str(len(columns)) + "," + stackCols + ")")\
                   .select(pivotCol, "col0", "col1")
            final_df = df_1.groupBy(col("col0")).pivot(pivotCol).agg(concat_ws("", collect_list(col("col1"))))\
                         .withColumnRenamed("col0", pivotCol)
            return final_df

        except Exception as e:

            print('TransposeDF failed!')
            print(e)

    def summary_statistic_1(self):

        dataframe = self.data.summary()


        dataframe = self.TransposeDF(dataframe, dataframe.columns, "Summary")
        dataframe = dataframe.withColumnRenamed("Summary", "Features")
        return dataframe




    def calculate_summary_statistics(self,data):

        dataframe = data.summary()
        print(type(dataframe))

        dataframe = self.TransposeDF(dataframe, dataframe.columns, "Summary")
        dataframe = dataframe.withColumnRenamed("Summary", "Features")



        num_unique_spark_df = data.agg(*(countDistinct(col(c)).alias(c) for c in data.columns))
        num_unique_spark_df = self.TransposeDF(num_unique_spark_df, num_unique_spark_df.columns, "ID")
        num_unique_spark_df = num_unique_spark_df.withColumnRenamed("ID", "Features").withColumnRenamed("2240", "Num_Unique")
        dataframe = dataframe.join(num_unique_spark_df,on='Features',how='left')



        dtype_spark_df = self.spark.createDataFrame(data.dtypes).withColumnRenamed("_1", "Features").withColumnRenamed("_2", "dtypes")
        dataframe = dataframe.join(dtype_spark_df,on='Features',how='left')




        isna_sum_spark_df = data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns])

        isna_sum_spark_df = self.TransposeDF(isna_sum_spark_df, isna_sum_spark_df.columns, isna_sum_spark_df.columns[0])
        isna_sum_spark_df = isna_sum_spark_df.withColumnRenamed(isna_sum_spark_df.columns[0], "Features").withColumnRenamed("0", "num_nan")

        dataframe = dataframe.join(isna_sum_spark_df,on='Features',how='left')


        return dataframe



    def get_summary_statistics(self):


        data = self.read_data()
        result = self.calculate_summary_statistics(data)

        return result


if __name__ == '__main__':

    obj = SummaryStatistics(spark,'src/data/kc_house_data.csv')
    df = obj.get_summary_statistics()
    df.coalesce(1).write.mode('overwrite').option('header','true').csv('summary_statistics.csv')
    print('Done!')


