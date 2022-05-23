// Databricks notebook source
// MAGIC 
// MAGIC %md
// MAGIC ### Ecommerce Revenue Prediction 
// MAGIC Predicting spending of customer based on features like time spent on website, average time of sessions with personal stylists from the store etc.,
// MAGIC Hence helping business to make improvements in gaining more loyal customers.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Loading Data

// COMMAND ----------

import org.apache.spark.sql.Encoders

// defining the schema
case class Customer(Email: String,
                     Avatar: String,
                     Avg_Session_Length: Double,
                     Time_on_App: Double,
                     Time_on_Website: Double,
                     Length_of_Membership: Double,
                     Yearly_Amount_Spent: Double)

val CustomerSchema = Encoders.product[Customer].schema

val CustomerDF = spark.read.schema(CustomerSchema).option("header", "true").csv("/FileStore/tables/ecommerce.csv")

display(CustomerDF)

// COMMAND ----------

CustomerDF.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Data Summary

// COMMAND ----------

CustomerDF.select("Avg_Session_Length","Time_on_App", "Time_on_Website", "Length_of_Membership", "Yearly_Amount_Spent").describe().show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Creating temporary view to query the dataframe

// COMMAND ----------

CustomerDF.createOrReplaceTempView("CustomerData")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from CustomerData

// COMMAND ----------

// MAGIC %md
// MAGIC #### EDA

// COMMAND ----------

// DBTITLE 1,Relationship between all features
// MAGIC %sql
// MAGIC select Email, Avatar, Avg_Session_Length, Time_on_App, Time_on_Website, Length_of_Membership, Yearly_Amount_Spent from CustomerData

// COMMAND ----------

// DBTITLE 1,Types of Fashion
// MAGIC %sql
// MAGIC 
// MAGIC select Avatar as Fashion, count(Avatar) from CustomerData group by Avatar

// COMMAND ----------

// DBTITLE 1,Amount spent on an yearly basis
// MAGIC %sql
// MAGIC 
// MAGIC select Yearly_Amount_Spent from CustomerData

// COMMAND ----------

// DBTITLE 1,Amount spent based on time used on app
// MAGIC %sql
// MAGIC 
// MAGIC select Yearly_Amount_Spent, Time_on_App from CustomerData

// COMMAND ----------

// DBTITLE 1,Amount spent based on time used on website
// MAGIC %sql
// MAGIC 
// MAGIC select Yearly_Amount_Spent, Time_on_Website from CustomerData

// COMMAND ----------

// DBTITLE 1,Amount spent based on average length of session with personal stylist
// MAGIC %sql
// MAGIC 
// MAGIC select Yearly_Amount_Spent, Avg_Session_Length from CustomerData

// COMMAND ----------

// MAGIC %md
// MAGIC #### Linear Regression model

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}

//Concatenating columns into vector
var FeatureCol = Array("Email", "Avatar")

//Converting string values to indexes for categorical features
val index = FeatureCol.map { colName =>
  new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
}

val lrpipeline = new Pipeline().setStages(index)      

val FinaleCusotmerDF = lrpipeline.fit(CustomerDF).transform(CustomerDF)


// COMMAND ----------

FinaleCusotmerDF.printSchema()

// COMMAND ----------

FinaleCusotmerDF.show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Splitting the data

// COMMAND ----------

val split = FinaleCusotmerDF.randomSplit(Array(0.7, 0.3))
val trainData = split(0)
val testData = split(1)
val trainRows = trainData.count()
val testRows = testData.count()
println("Training Rows: " + trainRows + " Testing Rows: " + testRows)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Data Preproccessing

// COMMAND ----------

// DBTITLE 1,Combining categorical variables into a single feature using Vector Assembler
val assembler = new VectorAssembler().setInputCols(Array("Email_indexed", "Avatar_indexed", "Avg_Session_Length", "Time_on_App", "Time_on_Website", "Length_of_Membership")).setOutputCol("features")

val trainingData = assembler.transform(trainData).select($"features", $"Yearly_Amount_Spent".alias("label"))

trainingData.show()

// COMMAND ----------

val testingData = assembler.transform(testData).select($"features", $"Yearly_Amount_Spent".alias("trueLabel"))

testingData.show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Training the model

// COMMAND ----------

val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
val model = lr.fit(trainingData)
println("Model training complete")

// COMMAND ----------

// MAGIC %md
// MAGIC #### Testing the model

// COMMAND ----------

val prediction = model.transform(testingData)
val predicted = prediction.select("features", "trueLabel", "prediction" )
predicted.show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Model Performance

// COMMAND ----------

predicted.createOrReplaceTempView("CustomerData")

// COMMAND ----------

// MAGIC %sql
// MAGIC 
// MAGIC select prediction, trueLabel from CustomerData

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val evaluation = new RegressionEvaluator().setLabelCol("trueLabel").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluation.evaluate(prediction)
println("Root Mean Square Error (RMSE): " + (rmse))
