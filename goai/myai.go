package main

import (
	plotcode "github.com/scofild429/goai/goai/plot"
)

func main() {

	// singlenode.Single_node_iris(true)
	// mpicode.Mpi_iris_Allreduce()
	// mpicode.Mpi_iris_SendRecv()
	// mpicode.Mpi_images_Allreduce()
	// mpicode.Mpi_images_SendRecv()
	plotcode.PlotIrisSpeedup()
	plotcode.PlotIntelImgageSpeedup()
}
