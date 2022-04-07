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


    def read_data(self):

        try:

            data = self.spark.read.csv(self.file_path,inferSchema=True,header=True,encoding='utf-8',sep='\t')
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


    def calculate_summary_statistics(self,data):

        dataframe = data.summary()

        dataframe = self.TransposeDF(dataframe, dataframe.columns, "Summary")
        dataframe = dataframe.withColumnRenamed("Summary", "Features")


        num_unique_spark_df = data.agg(*(countDistinct(col(c)).alias(c) for c in data.columns))
        num_unique_spark_df = self.TransposeDF(num_unique_spark_df, num_unique_spark_df.columns, "ID")
        num_unique_spark_df = num_unique_spark_df.withColumnRenamed("ID", "Features").withColumnRenamed("2240", "Num_Unique")
        dataframe = dataframe.join(num_unique_spark_df,on='Features',how='left')


        type_list = [str(data.schema[i].dataType) for i in data.columns]
        type_spark_df = self.spark.createDataFrame([(l,) for l in type_list], ['Type'])

        dataframe = dataframe.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
        type_spark_df = type_spark_df.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))

        dataframe = dataframe.join(type_spark_df, dataframe.row_idx == type_spark_df.row_idx).drop("row_idx")




        isna_sum_spark_df = data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns])

        isna_sum_spark_df = self.TransposeDF(isna_sum_spark_df, isna_sum_spark_df.columns, "ID")
        isna_sum_spark_df = isna_sum_spark_df.withColumnRenamed("ID", "Features").withColumnRenamed("0", "num_nan")

        dataframe = dataframe.join(isna_sum_spark_df,on='Features',how='left')


        return dataframe



    def get_summary_statistics(self):


        data = self.read_data()
        result = self.calculate_summary_statistics(data)

        return result




if __name__ == "__main__":
    obj = SummaryStatistics(spark,'src/data/marketing_campaign.csv')
    df = obj.get_summary_statistics()
    df.coalesce(1).write().mode('overwrite').options('header','true').csv('summary_statistics.csv')
    print('Done!')
