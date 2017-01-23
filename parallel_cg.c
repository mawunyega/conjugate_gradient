#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define EPSILON 1.0e-6
#define MAX_ITERATION 10
#define rows 256
#define cols 256


// add time functions !!!!!!!

void initialize(float vector[], char filename[], int col);

void matVec(float local_matvec[], float local_matrixA[], int local_row, float local_vectorX[]);

void residual(float local_vector[], float local_vectorB[], int local_row, float local_matvec[]);

float vecVec(float vect1[],float vect2[], int local_row);

void scalarVec(float temp[], float vect[], float a, int local_row);

void vecAdd(float vect[], float vect1[], float vect2[], int local_row);

void vecSub(float vect[], float vect1[], float vect2[], int local_row);

void conjugrad(float local_matrixA[],float local_vectorB[],float local_vectorX[], int local_row, int myrank);

void printer(float vect[], int rows1, int col);


int main(int argc, char* argv[])
{	

 clock_t t;

	FILE *reader;

	FILE *reader1;

	FILE *reader2;



	float *matrixA;

	float *vectorB;

	float *local_matrixA;

	float *local_vectorX;

	float *local_vectorB;
	
	double start_time, finish_time,clock_time_taken;
	




	int myrank, p, local_row, col,i;

	col = 1;

	char filename[] ="matrixA_256X256.txt";

	char filename1[] ="vectorb_256X1.txt";

	char filename2[] ="X0_256X1.txt";


	
	
	MPI_Status status;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &p);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    start_time = MPI_Wtime();
    
    t = clock();
	

    local_row = rows / p; // here must be changed to a cyclic data distribution
    
    //printf("my rows are %d\n", local_row);



    if((local_matrixA = malloc((local_row*cols)*sizeof(float)))==NULL)
    {
    	printf("can't allocate memory for local matrix\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


   
	if((local_vectorX = malloc((rows*col)*sizeof(float)))==NULL)
	{
		printf("can't allocate memory for vector X\n");

		MPI_Abort(MPI_COMM_WORLD,1);
	}
		


    if((local_vectorB = malloc (local_row*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for local vector B\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }



    if(myrank == 0)
    {
    	if((matrixA = malloc((rows*cols)*sizeof(float)))== NULL)
    	{
    		printf("can't allocate memory for matrix\n");

    		MPI_Abort(MPI_COMM_WORLD,1);
    	}
    	else
    	{
    		initialize(matrixA, filename, cols);
    	}

    }


    if(myrank == 0)
    {
    	if((vectorB = malloc(rows*sizeof(float)))==NULL)
    	{
    		printf("can't allocate memory for vector B\n");

    		MPI_Abort(MPI_COMM_WORLD,1);
    	}
    	else
    	{
    		
    		initialize(vectorB, filename1, col);

    		
    		initialize(local_vectorX, filename2, col);

    	}

    }


	//broadcast x0 to every processor

    MPI_Bcast(local_vectorX,rows*col,MPI_FLOAT,0,MPI_COMM_WORLD);
    
    


    //spilt matrix A

    MPI_Scatter(matrixA,local_row*cols,MPI_FLOAT,local_matrixA,local_row*cols, MPI_FLOAT,0,MPI_COMM_WORLD);


    MPI_Scatter(vectorB,local_row,MPI_FLOAT,local_vectorB,local_row, MPI_FLOAT,0,MPI_COMM_WORLD);

  

    conjugrad(local_matrixA, local_vectorB, local_vectorX, local_row, myrank);

	MPI_Barrier(MPI_COMM_WORLD);
  	finish_time = MPI_Wtime();
  	t = clock() - t;
  	clock_time_taken = ((double)t)/CLOCKS_PER_SEC;
   if(myrank == 0)
   {
   	printf("matrix size : %d\n", rows*cols );
   	printf("average mpi execution time in seconds: %f\n", (finish_time - start_time) );
   	printf("average clock execution time in seconds: %lu\n", clock_time_taken );
   	
   }
    
   
    if(myrank == 0)
    {
    	free(matrixA);

    	free(vectorB);
    }
   

    free(local_matrixA);

    free(local_vectorX);

    free(local_vectorB);

	MPI_Finalize();

	return 0;

}


void initialize(float vector[], char filename[], int col)
{

	FILE *reader;

	int i,j,q;

	MPI_Status status;

	reader = fopen(filename, "r");

	if(reader != NULL)
	{
		for(i = 0; i < rows; i++)
		{
			for(j = 0; j < col; j++)
			{
				fscanf(reader,"%f%*c",&vector[i*col+j]);
			}
		}

		fclose(reader);
	}
    else
	{
		printf("Could not open file\n");
	}
	
}


void matVec(float local_matvec[], float local_matrixA[], int local_row, float local_vectorX[])
{

	int i,j;

	for(i = 0; i < local_row; i++)
	{
		local_matvec[i] = 0.0;

		for(j = 0; j < cols; j++)
		{
			local_matvec[i] += local_matrixA[i*cols+j] * local_vectorX[j];

		}
		
	}	

}


void residual(float local_vector[], float local_vectorB[], int local_row, float local_matvec[])
{
	int i;

	
	for(i=0; i < local_row; i++)
	{
		
		local_vector[i] = local_vectorB[i] - local_matvec[i];
	}

}


float vecVec(float vect1[],float vect2[], int local_row)
{
	int i;

	float sum;

	sum = 0.0;

	for (i=0; i < local_row; i++)
	{
		sum += vect1[i] * vect2[i] ;
	}

	return sum;

}

void scalarVec(float temp[], float vect[], float a, int local_row)
{
	int i;

	for(i=0; i < local_row; i++)
	{
		temp[i] = vect[i] * a;
	}

}


void vecAdd(float vect[], float vect1[], float vect2[], int local_row)
{
	int i;

	for (i=0; i < local_row; i++)
	{
		vect[i] = vect1[i] + vect2[i] ;
	}	

}

void vecSub(float vect[], float vect1[], float vect2[], int local_row)
{
	int i;

	for (i=0; i < local_row; i++)
	{
		vect[i] = vect1[i] - vect2[i] ;
	}	

}

void conjugrad(float local_matrixA[],float local_vectorB[],float local_vectorX[], int local_row, int myrank)
{
	float *local_matvec;

	float *local_vectorR;

	float *local_vectorP;

	float *vectorP;

	float *vectorR;

	float *temp_x;

	float *temp_r;

	float *temp_p;

	

	float rsold, temp_rsold, temp_alpha, alpha, temp_beta,beta, store;

	int i,j,k,printcol;

	printcol = 1;

	
	MPI_Status status;


	if((local_matvec = malloc (local_row*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for local vector for matVec\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }

    if((local_vectorR = malloc (local_row*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for local vector R\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


    if((local_vectorP = malloc (local_row*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for local vector R\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


    if((vectorP = malloc (rows*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for vector P\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


    if((vectorR = malloc ((rows*cols)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for vector R\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


    if((temp_x = malloc (rows*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for temp local vector X\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }

    if((temp_r = malloc (local_row*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for temp local vector R\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }

    if((temp_p = malloc (local_row*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for temp local vector P\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


	matVec(local_matvec,local_matrixA, local_row, local_vectorX);
	
	


	residual(local_vectorR, local_vectorB, local_row, local_matvec);

	

	residual(local_vectorP, local_vectorB, local_row, local_matvec);

	/*if(myrank == 0)
    {
    	printer(local_vectorP, local_row, k);
    }*/

	temp_rsold = vecVec(local_vectorR, local_vectorR,  local_row);

	MPI_Allreduce(&temp_rsold,&rsold,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

	//printf("rsold is %f\n", rsold);

	

	

	for(k=0; k < rows; k++)
	{

			MPI_Allgather(local_vectorP, local_row, MPI_FLOAT, vectorP , local_row, MPI_FLOAT,MPI_COMM_WORLD);
		
			//ap =a*p
			matVec(local_matvec,local_matrixA, local_row, vectorP); 

			
    		//printer(local_matvec, local_row, 1);
  

			//alpha = rsold / (p' *ap)

			temp_alpha = vecVec(local_vectorP, local_matvec,  local_row);

			//printf("result b4 is %f \n", temp_alpha);


			MPI_Allreduce(&temp_alpha,&alpha,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);


			alpha = rsold / alpha;

			//printf("result temp_alpha is %f \n", temp_alpha);

			 

			scalarVec(temp_x, vectorP, alpha, rows);

			vecAdd(local_vectorX, local_vectorX, temp_x, rows);
			//if(myrank == 0)
				//printer(local_vectorX, rows,1);
			


			scalarVec(temp_r, local_matvec, alpha, local_row);

			vecSub(local_vectorR, local_vectorR, temp_r, local_row);

			// everything fine till here

			

			//MPI_Allgather(local_vectorR, local_row, MPI_FLOAT, vectorR , local_row, MPI_FLOAT,MPI_COMM_WORLD);

			temp_beta = vecVec(local_vectorR, local_vectorR,  local_row);


			MPI_Allreduce(&temp_beta,&beta,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);


			if(sqrt(beta) < EPSILON)
			{
				break;
			}

			store = beta / rsold;

			//printf("result beta is %f \n", store);

			scalarVec(temp_p, local_vectorP, store, local_row);

			vecAdd(local_vectorP, local_vectorR, temp_p, local_row);
			
			

			rsold = beta;
			
			//printf("hello\n");


	}
	
	 if(myrank == 1)  
		printer(local_vectorX, rows, 1);

    			 


	free(local_matvec);

	free(local_vectorR);

	free(local_vectorP);

	free(vectorP);

	free(vectorR);

	free(temp_x);

	free(temp_r);

	free(temp_p);


}





void printer(float vect[], int rows1, int col)
{
	int i, j;

	for(i=0; i < rows1; i++)
	{
		for(j=0; j < col; j++)
		{
			printf("%f   ", vect[i*col+j]);

			if(j == (col - 1)) 
				printf("\n");
		}
		
	}

}
