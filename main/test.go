package main

import (
	"github.com/victorddiniz/fitness_function_analysis/functions"
	"fmt"
	"github.com/victorddiniz/fitness_function_analysis/iohandlers"
	"github.com/victorddiniz/fitness_function_analysis/es"
)

func main() {
	var bestInd * es.Individual
	bestValidation := 0.0

	for i := 0; i < 10; i++ {
		population := es.NewPopulation(1, 1, 100000, "../datasets/sun.txt")
		ioHandler := iohandlers.GetInstance()
		indRun, valueTraining := population.Run()
		in, out := ioHandler.GetKLagValidationSet(indRun.GetLag())
		value := indRun.Fitness(in, out)

		if bestInd == nil || value > bestValidation {
			bestInd = indRun
			bestValidation = value
		}
		fmt.Println(value, valueTraining)
	}

	ioHandler := iohandlers.GetInstance()
	in, target := ioHandler.GetKLagTestSet(bestInd.GetLag())
	obs := bestInd.Predict(in)
	fmt.Println(functions.MSE(target, obs))
}
