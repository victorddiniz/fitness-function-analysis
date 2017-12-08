package functions

import (
	"math"
)

// Sigmoid ...
func Sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
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
	return sum / float64(N)
}

// MAPE ...
func MAPE(target, obs []float64) float64 {
	N := len(target)
	sum := 0.0
	for i := 0; i < N; i++ {
		e := error(target[i], obs[i])
		sum += math.Abs(e / target[i])
	}
	return sum / float64(N)
}

// Theil ...
func Theil(target, obs []float64) float64 {
	N := len(target)
	mse := MSE(target, obs) * float64(N)
	randomWalk := 0.0
	for i := 0; i < N-1; i++ {
		value := target[i] - target[i+1]
		randomWalk += (value * value)
	}
	return mse / randomWalk
}

// POCID ...
func POCID(target, obs []float64) float64 {
	N := len(target)
	sum := 0.0
	for i := 1; i < N; i++ {
		factor := (target[i] - target[i-1]) * (obs[i] - obs[i-1])
		if factor > 0 {
			sum++
		}
	}
	return 100.0 * (sum / float64(N))
}

// ARV ...
func ARV(target, obs []float64) float64 {
	N := len(target)
	tBar := 0.0
	sum := 0.0
	mse := MSE(target, obs) * float64(N)
	for i := 0; i < N; i++ {
		tBar += (target[i])
	}
	tBar = tBar / float64(N)
	for i := 0; i < N; i++ {
		e := (obs[i] - tBar)
		sum += (e * e)
	}
	return mse / sum
}
