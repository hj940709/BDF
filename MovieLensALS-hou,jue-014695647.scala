import java.util.Random

import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.io.Source

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object MovieLensALS {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)	

    if (args.length != 1) {
      println("Usage: sbt/sbt package \"run movieLensHomeDir\"")
      exit(1)
    }

    // set up environment

    val jarFile = "target/scala-2.10/movielens-als_2.10-0.0.jar"
    val sparkHome = "/root/spark"
    val master = Source.fromFile("/root/spark-ec2/cluster-url").mkString.trim
    val masterHostname = Source.fromFile("/root/spark-ec2/masters").mkString.trim
    val conf = new SparkConf()
      .setMaster(master)
      .setSparkHome(sparkHome)
      .setAppName("MovieLensALS")
      .set("spark.executor.memory", "8g")
      .setJars(Seq(jarFile))
    val sc = new SparkContext(conf)

    // load ratings and movie titles

    val movieLensHomeDir = "hdfs://" + masterHostname + ":9000" + args(0)
	
	//read rating data file
    val ratings = sc.textFile(movieLensHomeDir + "/ratings.dat").map { line =>
      val fields = line.split("::")
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }
	//read movies data file
    val movies = sc.textFile(movieLensHomeDir + "/movies.dat").map { line =>
      val fields = line.split("::")
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }.collect.toMap

    val numRatings = ratings.count //count the number of rating records
    val numUsers = ratings.map(_._2.user).distinct.count //count the number of users
    val numMovies = ratings.map(_._2.product).distinct.count //count the number of movies

    println("Got " + numRatings + " ratings from "
      + numUsers + " users on " + numMovies + " movies.")

    // sample a subset of most rated movies for rating elicitation

    val mostRatedMovieIds = ratings.map(_._2.product) // extract movie ids
                                   .countByValue      // count ratings per movie
                                   .toSeq             // convert map to Seq
                                   .sortBy(- _._2)    // sort by rating count
                                   .take(50)          // take 50 most rated
                                   .map(_._1)         // get their ids
    val random = new Random(0) //random number generator
	// select movies arbitrarily
    val selectedMovies = mostRatedMovieIds.filter(x => random.nextDouble() < 0.2)
                                          .map(x => (x, movies(x)))
                                          .toSeq

    // elicitate ratings

    val myRatings = elicitateRatings(selectedMovies)
    val myRatingsRDD = sc.parallelize(myRatings, 1)

    // split ratings into train (60%), validation (20%), and test (20%) based on the 
    // last digit of the timestamp, add myRatings to train, and cache them

    val numPartitions = 20
    val training = ratings.filter(x => x._1 < 6) 
                          .values 
                          .union(myRatingsRDD) 
                          .repartition(numPartitions)
                          .persist 
	//filter out the last digit of timestamp smaller than 6
	//extract rating records
	//mix with myRatings
	//repartition and cache training data
	
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8) 
                            .values
                            .repartition(numPartitions)
                            .persist
	//filter out the last digit of timestamp bigger than 6 but smaller than 8
	//extract rating records
	//repartition and cache validation data
	
    val test = ratings.filter(x => x._1 >= 8).values.persist
	//filter out the last digit of timestamp bigger than 8 
	//extract rating records
	//repartition and cache test data
	
	
	//count the number of data points for training, validation and test data
    val numTraining = training.count
    val numValidation = validation.count
    val numTest = test.count

    println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest)

    // train models and evaluate them on the validation set

    val ranks = List(8, 12)
    val lambdas = List(0.1, 10.0)
    val numIters = List(10, 20)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
	// iteration to find model with lowest validation rmse
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda) //ALS training
      val validationRmse = computeRmse(model, validation, numValidation)
	  // compute rmse for validation with previously trained model
      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = " 
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    // evaluate the best model on the test set

    val testRmse = computeRmse(bestModel.get, test, numTest)

    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")

    // create a naive baseline and compare it with the best model

    val meanRating = training.union(validation).map(_.rating).mean
	//calculate the mean rating of training and validation data
    val baselineRmse = math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating))
                                     .reduce(_ + _) / numTest)
	//calculate baselin rmse
	//sum((x-mean)^2)/testnum
    val improvement = (baselineRmse - testRmse) / baselineRmse * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    // make personalized recommendations

    val myRatedMovieIds = myRatings.map(_.product).toSet
	//filter away movies movies
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    val recommendations = bestModel.get
                                   .predict(candidates.map((0, _)))
                                   .collect
                                   .sortBy(- _.rating)
                                   .take(50)

    var i = 1
    println("Movies recommended for you:")
    recommendations.foreach { r =>
      println("%2d".format(i) + ": " + movies(r.product))
      i += 1
    }

    // clean up

    sc.stop();
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long) = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
                                           .join(data.map(x => ((x.user, x.product), x.rating)))
                                           .values
	//set user and product as keys and rating as value
	//extract values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
	//rmse sum((x1-x2)^2)/n
  }
  
  /** Elicitate ratings from command-line. */
  def elicitateRatings(movies: Seq[(Int, String)]) = {
    val prompt = "Please rate the following movie (1-5 (best), or 0 if not seen):"
    println(prompt)
    val ratings = movies.flatMap { x =>
      var rating: Option[Rating] = None
      var valid = false
      while (!valid) {
        print(x._2 + ": ")
        try {
          val r = Console.readInt
          if (r < 0 || r > 5) {
            println(prompt)
          } else {
            valid = true
            if (r > 0) {
              rating = Some(Rating(0, x._1, r))
            }
          }
        } catch {
          case e: Exception => println(prompt)
        }
      }
      rating match {
        case Some(r) => Iterator(r)
        case None => Iterator.empty
      }
    }
    if(ratings.isEmpty) {
      error("No rating provided!")
    } else {
      ratings
    }
  }
}
/*
In this sample code, there are obviously many spark actions and transfermations.
Most of them is used to formatize data presentation so that further calculation can proceed. 
For example, the author set user and movie as keys and rating as value. 
Then, values can be extracted directly with ".values".
Also, there are actions are used to calculating things such as rmse or 
filter out data according to the last digit of timestamp.

If it comes to actual ASL algorithm, author called embedded function. This is the core of collebrative filtering.
Some number countings have also be called, but they seems to used to simply showing number information.
I doubt such counting has actual meaning for algorithm.
*/