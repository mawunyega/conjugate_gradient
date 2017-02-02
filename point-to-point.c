#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#define EPSILON 1.0e-6
#define ROWS 4
#define COLS 4
#define COL 1


void initialize(float vector[], 
  char filename[], 
  int colunm_num);

void matVec(float local_matvec[], 
  float local_matrixA[], 
  int local_row, 
  float local_vectorX[]);

void residual(float local_vector[], 
  float local_vectorB[], 
  int local_row, 
  float local_matvec[]);

float vecVec(float vect1[],
  float vect2[], 
  int local_row);

void scalarVec(float temp[], 
  float vect[], 
  float scalar, 
  int local_row);

void vecAdd(float vect[], 
  float vect1[], 
  float vect2[], 
  int local_row);

void vecSub(float vect[], 
  float vect1[], 
  float vect2[], 
  int local_row);
  
void conjugrad(float local_matrixA[],
  float local_vectorB[],
  float local_vectorX[], 
  int local_row, 
  int myrank, int procsnum);
void scatterRowwise(float *local_matrix, 
  char *filename, int local_row, int myrank, 
  int procsnum, int col);
void broadcastVector(float *local_vectorX,
    int myrank, int procsnum, int cnt);
float allSum(int myrank,int procsnum, float temp_rsold, float rsold);
void allGather(float *vect, float *vect1,int local_row,int myrank,int procsnum);
void printer(float vect[], int rows1, int colunm_num);
int main(int argc, char* argv[])
{ 
clock_t t;
float *matrixA,
      *vectorB,
      *local_matrixA,
      *temp,
      *local_vectorX,
      *local_vectorB;
double start_time, 
       finish_time,
       clock_time_taken;
int myrank, 
    procsnum,
    i,
    j,
    dest, 
    local_row;
MPI_Status status;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &procsnum);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Barrier(MPI_COMM_WORLD);
start_time = MPI_Wtime();
t = clock();
local_row = ROWS / procsnum; 
if((local_matrixA = malloc((local_row*COLS)*sizeof(float)))==NULL) /////check these!!!!
{
printf("can't allocate memory for local matrix\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}

if((local_vectorX = malloc((ROWS*COL)*sizeof(float)))==NULL)
{
  printf("can't allocate memory for vector X\n");
  MPI_Abort(MPI_COMM_WORLD,1);
}
if((local_vectorB = malloc ((local_row*COL)*sizeof(float))) == NULL)
{
    printf("can't allocate memory for local vector B\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}
if(myrank == 0)
{
  if((matrixA = malloc((ROWS*COLS)*sizeof(float)))== NULL)
      { 
        printf("can't allocate memory for matrix\n");
        MPI_Abort(MPI_COMM_WORLD,1);
      }
      
}
if(myrank == 0)
{
      if((vectorB = malloc((ROWS*COL)*sizeof(float)))==NULL)
      {
        printf("can't allocate memory for vector B\n");
        MPI_Abort(MPI_COMM_WORLD,1);
      }
      else
      {
        //initialize(vectorB, argv[2], COL);
        initialize(local_vectorX, argv[3], COL);
      }
}
//MPI_Bcast(local_vectorX,ROWS*COL,MPI_FLOAT,0,MPI_COMM_WORLD);
broadcastVector(local_vectorX,myrank, procsnum, ROWS);

//MPI_Scatter(matrixA,local_row*COLS,MPI_FLOAT,local_matrixA,local_row*COLS, MPI_FLOAT,0,MPI_COMM_WORLD);

scatterRowwise(local_matrixA, argv[1], local_row, myrank, procsnum, COLS); 

scatterRowwise(local_vectorB, argv[2], local_row, myrank, procsnum, COL); 

conjugrad(local_matrixA, local_vectorB, local_vectorX, local_row, myrank, procsnum);
MPI_Barrier(MPI_COMM_WORLD);
finish_time = MPI_Wtime();
t = clock() - t;
clock_time_taken = ((double)t)/CLOCKS_PER_SEC; 


/*if(myrank == 0)
{
  printer(local_matrixA,local_row,COLS );
}*/

if(myrank == 0)
{
  printf("matrix size : %d\n", ROWS*COLS );
    printf("average mpi execution time in seconds: %f\n", (finish_time - start_time) );
    printf("average clock execution time in seconds: %f\n", clock_time_taken );   
}

free(local_matrixA);
free(temp);
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
  for(i = 0; i < ROWS; i++)
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
void scatterRowwise(float *local_matrix, char *filename, int local_row, int myrank, int procsnum, int col)
{
  FILE *reader;
  int dest,i,j;
  float *temp;
  MPI_Status status;
  if((temp = malloc((local_row*col)*sizeof(float)))==NULL)
  {
    printf("can't allocate memory for local matrix\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  if(myrank == 0)
  {
    reader = fopen(filename, "r");
  if (reader != NULL)
  {
    for(i = 0; i < local_row; ++i)
    {
       for(j=0; j < col; ++j)
       {
         fscanf(reader, "%f%*c", &local_matrix[i*col+j]);
       }
    }
    
    for (dest = 1; dest < procsnum; ++dest) 
    {
      for (i = 0; i < local_row; ++i)
      {
        for(j=0; j < col; ++j)
        {
          fscanf(reader, "%f%*c", &temp[i*col+j]);
        }
      }
      MPI_Send(temp,local_row*col, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);  
    }
    fclose(reader);
  }else
      printf("Could not open matrix file. \n");
 }else 
   {
      MPI_Recv(local_matrix, local_row*COLS, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
   }

}
void broadcastVector(float *local_vectorX, int myrank, int procsnum, int cnt)
{
  int dest,i;
  MPI_Status status;
 if(myrank == 0)
 {
  for(i = 1; i < procsnum; ++i)
  {
    MPI_Send(local_vectorX, cnt,MPI_FLOAT,i,0, MPI_COMM_WORLD);
  }
 }else
 {
   MPI_Recv(local_vectorX, cnt,MPI_FLOAT,0,0, MPI_COMM_WORLD, &status);
 }
}
void matVec(float local_matvec[], float local_matrixA[], int local_row, float local_vectorX[])
{
int i,j;
for(i = 0; i < local_row; i++)
{
  local_matvec[i] = 0.0;
  for(j = 0; j < COLS; j++)
  {
    local_matvec[i] += local_matrixA[i*COLS+j] * local_vectorX[j];
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
void scalarVec(float temp[], float vect[], float scalar, int local_row)
{
int i;
for(i=0; i < local_row; i++)
{
  temp[i] = vect[i] * scalar;
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
float allSum(int myrank,int procsnum, float temp_rsold, float rsold)
{
  int dest, cnt;
  cnt = 1;
  MPI_Status status;
  if(myrank == 0)
  {
    rsold = temp_rsold;
    for(dest = 1; dest < procsnum; ++dest)
    {
      MPI_Recv(&temp_rsold,1,MPI_FLOAT,dest,3,MPI_COMM_WORLD,&status);
      rsold += temp_rsold;
    }

  }else
    {
      MPI_Send(&temp_rsold,1,MPI_FLOAT,0,3,MPI_COMM_WORLD);
    }

    
  return rsold;
  
}
void allGather(float *vect, float *vect1,int local_row,int myrank,int procsnum)
{
  int dest,i,index, rank;

  float *temp_localP;

  if((temp_localP = malloc ((local_row*COL)*sizeof(float))) == NULL)
  {
    printf("can't allocate memory for local vector for P\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  index = local_row;

  MPI_Status status;

  if(myrank == 0)
  {
    for(i=0; i < local_row; ++i)
    {
      vect[i] = vect1[i];
    }
    for(dest = 1; dest < procsnum; ++dest)
    {
      MPI_Recv(temp_localP, local_row, MPI_FLOAT, dest, 4, MPI_COMM_WORLD, &status);
      for(i = 0; i < local_row; ++i)
      {
        vect[index] = temp_localP[i];
        index++;
      }
      //printf("i got %f and %f from %d\n",local_vectorP[0], local_vectorP[1], dest );
    }
  }
  else
  {
    MPI_Send(vect1,local_row,MPI_FLOAT,0,4,MPI_COMM_WORLD);
  }
free(temp_localP);
   
}
void conjugrad(float local_matrixA[],float local_vectorB[],float local_vectorX[], int local_row, int myrank, int procsnum)
{
float *local_matvec,
      *local_vectorR,
      *local_vectorP,
      *temp_localP,
      *vectorP,
      *vectorR,
      *temp_x,
      *temp_r,
      *temp_p,
      rsold,
      rsold1,
      temp_rsold,
      temp_alpha,
      alpha,
      alpha1,
      temp_beta,
      beta,
      beta1,
      store;
int   i,j,k;
MPI_Status status;
if((local_matvec = malloc ((local_row*COL)*sizeof(float))) == NULL)
{
  printf("can't allocate memory for local vector for matVec\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}
if((local_vectorR = malloc ((local_row*COL)*sizeof(float))) == NULL)
{
  printf("can't allocate memory for local vector R\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}
if((local_vectorP = malloc ((local_row*COL)*sizeof(float))) == NULL)
{
  printf("can't allocate memory for local vector R\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}
if((vectorP = malloc ((ROWS*COL)*sizeof(float))) == NULL)
{
  printf("can't allocate memory for vector P\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}
if((vectorR = malloc ((ROWS*COL)*sizeof(float))) == NULL)
{
  printf("can't allocate memory for vector R\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}
if((temp_x = malloc ((ROWS*COL)*sizeof(float))) == NULL)
{
  printf("can't allocate memory for temp local vector X\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}
if((temp_r = malloc ((local_row*COL)*sizeof(float))) == NULL)
{
  printf("can't allocate memory for temp local vector R\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}
if((temp_p = malloc ((local_row*COL)*sizeof(float))) == NULL)
{
  printf("can't allocate memory for temp local vector P\n");
      MPI_Abort(MPI_COMM_WORLD,1);
}




matVec(local_matvec,local_matrixA, local_row, local_vectorX);
residual(local_vectorR, local_vectorB, local_row, local_matvec);

residual(local_vectorP, local_vectorB, local_row, local_matvec);

temp_rsold = vecVec(local_vectorR, local_vectorR,  local_row);

//MPI_Allreduce(&temp_rsold,&rsold,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

rsold = allSum(myrank,procsnum, temp_rsold,rsold1);

broadcastVector(&rsold, myrank, procsnum, 1);


for(k=0; k < ROWS; k++)
{
  //printf("result rsold is %f \n", rsold); 
  //MPI_Allgather(local_vectorP, local_row, MPI_FLOAT, vectorP , local_row, MPI_FLOAT,MPI_COMM_WORLD);

  //printf("this is %d and my p are %f and %f \n", myrank, local_vectorP[0],local_vectorP[1]);

  allGather(vectorP, local_vectorP,local_row,myrank,procsnum);

  broadcastVector(vectorP, myrank, procsnum, ROWS);

  /*if(myrank == 0)
  {
    memcpy(local_vectorP, temp_localP, local_row);
  }*/

 //printf("this is %d and my p after are %f and %f \n", myrank, local_vectorP[0],local_vectorP[1]);

  /*if(myrank == 1)
  {
    printer(vectorP,ROWS,1);
  }*/
  
  matVec(local_matvec,local_matrixA, local_row, vectorP);

  //printf("this is %d and my matvec are %f and %f \n", myrank, local_matvec[0],local_matvec[1]);

  //printf("this is %d and my p are %f and %f \n", myrank, local_vectorP[0],local_vectorP[1]);
  temp_alpha = vecVec(local_vectorP, local_matvec,  local_row);

  //printf("result temp_alpha is %f \n", temp_alpha); 

  //printf("i am %d and i have result b4 is %f \n", myrank, temp_alpha);

  //MPI_Allreduce(&temp_alpha,&alpha,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
  alpha = allSum(myrank,procsnum, temp_alpha,alpha1);

  broadcastVector(&alpha, myrank, procsnum, 1);

  //printf("result alpha b4 is %f \n", alpha); 

  alpha = rsold / alpha;

  //printf("result alpha  is %f \n", alpha); 

  scalarVec(temp_x, vectorP, alpha, ROWS);
  vecAdd(local_vectorX, local_vectorX, temp_x, ROWS);

//printf("this is %d and my x are %f and %f \n", myrank, local_vectorX[0],local_vectorX[1]);

  scalarVec(temp_r, local_matvec, alpha, local_row);
  vecSub(local_vectorR, local_vectorR, temp_r, local_row);
  temp_beta = vecVec(local_vectorR, local_vectorR,  local_row);
  //MPI_Allreduce(&temp_beta,&beta,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

  //printf("result temp_beta  is %f \n", temp_beta); 

  beta = allSum(myrank,procsnum, temp_beta,beta1);

  broadcastVector(&beta, myrank, procsnum, 1);

  printf("result beta  is %f \n", beta); 
  if(sqrt(beta) < EPSILON)
  {
    break;
  }
  store = beta / rsold;
  scalarVec(temp_p, local_vectorP, store, local_row);
  vecAdd(local_vectorP, local_vectorR, temp_p, local_row);
  rsold = beta;
} 

if(myrank == 0)
  {
      printer(local_vectorX, ROWS, 1);
    
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

  for(i=0; i < rows1; ++i)
  {
    for(j=0; j < colunm_num; ++j)
    {
      printf("%f  ", vect[i*colunm_num+j]);

      if(j == (colunm_num - 1)) 
        printf("\n");
    }
    
  }

}