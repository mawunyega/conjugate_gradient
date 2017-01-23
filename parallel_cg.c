#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define EPSILON 1.0e-6
#define MAX_ITERATION 10
#define rows 1024
#define cols 1024
#define col 1


void initialize(float vector[], char filename[], int colunm_num);

void matVec(float local_matvec[], float local_matrixA[], int local_row, float local_vectorX[]);

void residual(float local_vector[], float local_vectorB[], int local_row, float local_matvec[]);

float vecVec(float vect1[],float vect2[], int local_row);

void scalarVec(float temp[], float vect[], float a, int local_row);

void vecAdd(float vect[], float vect1[], float vect2[], int local_row);

void vecSub(float vect[], float vect1[], float vect2[], int local_row);

void conjugrad(float local_matrixA[],float local_vectorB[],float local_vectorX[], int local_row, int myrank);

void printer(float vect[], int rows1, int colunm_num);


int main(int argc, char* argv[])
{	
clock_t t;
FILE *reader, 
     *reader1, 
     *reader2;
float *matrixA,
      *vectorB,
      *local_matrixA,
      *local_vectorX,
      *local_vectorB;
double start_time, 
       finish_time,
       clock_time_taken;
int myrank, 
    procsnum, 
    local_row;
MPI_Status status;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &procsnum);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Barrier(MPI_COMM_WORLD);
start_time = MPI_Wtime();
t = clock();
local_row = rows / procsnum; 
if((local_matrixA = malloc((local_row*cols)*sizeof(float)))==NULL) /////check these!!!!
{
	printf("can't allocate memory for local matrix\n");
    	MPI_Abort(MPI_COMM_WORLD,1);
}
if((local_vectorX = malloc((rows*col)*sizeof(float)))==NULL)
{
	printf("can't allocate memory for vector X\n");
	MPI_Abort(MPI_COMM_WORLD,1);
}
if((local_vectorB = malloc ((local_row*col)*sizeof(float))) == NULL)
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
    		initialize(matrixA, argv[1], cols);
    	}
}
if(myrank == 0)
{
    	if((vectorB = malloc((rows*col)*sizeof(float)))==NULL)
    	{
    		printf("can't allocate memory for vector B\n");
    		MPI_Abort(MPI_COMM_WORLD,1);
    	}
    	else
    	{
    		initialize(vectorB, argv[2], col);
    		initialize(local_vectorX, argv[3], col);
    	}
}
MPI_Bcast(local_vectorX,rows*col,MPI_FLOAT,0,MPI_COMM_WORLD);
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
   	printf("average clock execution time in seconds: %f\n", clock_time_taken );  	
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
void initialize(float vector[], char filename[], int col_num)
{
FILE *reader;
int i,j;
MPI_Status status;
reader = fopen(filename, "r");
if(reader != NULL)
{
	for(i = 0; i < rows; i++)
	{
		for(j = 0; j < col_num; j++)
		{
			fscanf(reader,"%f%*c",&vector[i*col_num+j]);
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

	int i,j,k;

	

	
	MPI_Status status;


	if((local_matvec = malloc ((local_row*col)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for local vector for matVec\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }

    if((local_vectorR = malloc ((local_row*col)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for local vector R\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


    if((local_vectorP = malloc ((local_row*col)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for local vector R\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


    if((vectorP = malloc ((rows*col)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for vector P\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


    if((vectorR = malloc ((rows*col)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for vector R\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }


    if((temp_x = malloc ((rows*col)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for temp local vector X\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }

    if((temp_r = malloc ((local_row*col)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for temp local vector R\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }

    if((temp_p = malloc ((local_row*col)*sizeof(float))) == NULL)
    {
    	printf("can't allocate memory for temp local vector P\n");

    	MPI_Abort(MPI_COMM_WORLD,1);
    }
	matVec(local_matvec,local_matrixA, local_row, local_vectorX);
	residual(local_vectorR, local_vectorB, local_row, local_matvec);
	residual(local_vectorP, local_vectorB, local_row, local_matvec);
	temp_rsold = vecVec(local_vectorR, local_vectorR,  local_row);
	MPI_Allreduce(&temp_rsold,&rsold,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	for(k=0; k < rows; k++)
	{
			MPI_Allgather(local_vectorP, local_row, MPI_FLOAT, vectorP , local_row, MPI_FLOAT,MPI_COMM_WORLD);
			matVec(local_matvec,local_matrixA, local_row, vectorP); 
			temp_alpha = vecVec(local_vectorP, local_matvec,  local_row);
			MPI_Allreduce(&temp_alpha,&alpha,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
			alpha = rsold / alpha;		 
			scalarVec(temp_x, vectorP, alpha, rows);
			vecAdd(local_vectorX, local_vectorX, temp_x, rows);
			scalarVec(temp_r, local_matvec, alpha, local_row);
			vecSub(local_vectorR, local_vectorR, temp_r, local_row);
			temp_beta = vecVec(local_vectorR, local_vectorR,  local_row);
			MPI_Allreduce(&temp_beta,&beta,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
			if(sqrt(beta) < EPSILON)
			{
				break;
			}
			store = beta / rsold;
			scalarVec(temp_p, local_vectorP, store, local_row);
			vecAdd(local_vectorP, local_vectorR, temp_p, local_row);
			rsold = beta;
	}				
	free(local_matvec);
	free(local_vectorR);
	free(local_vectorP);
	free(vectorP);
	free(vectorR);
	free(temp_x);
	free(temp_r);
	free(temp_p);
}
void printer(float vect[], int rows1, int colunm_num)
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
