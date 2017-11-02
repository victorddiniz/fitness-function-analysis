package functions

import (
	"math"
)

// Sigmoid ...
func Sigmoid(x float64) float64 {
	return 1.0/(1 + math.Exp(-x))
}

func error(target, obs float64) float64 {
	return target - obs
}

// MSE ...
func MSE(target, obs []float64) float64 {
	N := len(target)
	sum := 0.0
	for i := 0; i < N; i++ {
		e := error(target[i], obs[i])
		sum += e * e
	}
	return sum/float64(N)
}