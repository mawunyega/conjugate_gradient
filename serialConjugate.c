#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define EPSILON 1.0e-7


// change all float to float

void initializeMatrixA(float *matrix, int rows, int cols, FILE *reader);

void initializeVectB(float *vect, int rows1, FILE *reader);

void initializeGuessVec(float *vect, int rows1, FILE *reader);

float residual(float *vectb, float *vectx, int rows1);

float vecNorm(float *vectr, int rows1);

float vecVec(float *vect1,float *vect2, int rows1);

float matVec (float *matrix, int rows, int cols, float *vectr);

float scalarVec(float *vect, float a, int rows1);

float vecAdd(float *vect, float *vect1, int rows1);

float vecSub(float *vect, float *vect1, int rows1);

void printer(float *vect, int rows1);

int main(void)
{


	/* rows and matrix columns variables*/
	int rows, cols, rows1,cols1,niter, maxiter;

	float normb;

	maxiter = 3;

	niter = 0;

	



	/* vectors to store matrices for computations*/
	float *matrixA;

	float *vectorb;


	/*initial guess , residual, and arbitary vectors of cg method*/
	float *vectorX;

	float *vectorR;

	float *vectorP;

	float *tempvec;

	float *tempscalarvec;

	float *tempmatvec;



    	float alpha;

    	float beta;

    	float rho;
    	float store;

    	float tempstore1;


    	FILE *reader;

	
	



	reader = fopen("dimensions.txt", "r");

	if(reader != NULL)
	{
		fscanf(reader," %d %d %d %d\n", &rows, &cols, &rows1, &cols1); 

		fclose(reader);
	}

	else
	{
		printf("Could not open filekkkk\n");
	}

	printf("%d %d %d %d\n", rows, cols, rows1, cols1);


	
	if (cols != rows1)
	{
		printf("Matrix and vector dimensions do not match. Terminating program........\n");
		exit(0);

	}


	matrixA = malloc ((rows * cols)*sizeof(float) );

	vectorb = malloc ((rows1 * cols1)*sizeof(float) );

	vectorX = malloc (rows1*sizeof(float) );

	vectorR = malloc (rows1 *sizeof(float) );

	vectorP = malloc (rows1*sizeof(float) );
	
	tempvec = malloc (rows1*sizeof(float) );

	tempmatvec = malloc (rows1*sizeof(float) );
	
	tempscalarvec = malloc (rows1*sizeof(float) );
	

	initializeMatrixA(matrixA, rows, cols, reader);


	
	initializeVectB(vectorb, rows1, reader);


	
	initializeGuessVec(vectorX, rows1,reader);

	printf("i was previously here!!!\n");

	*tempmatvec = matVec(matrixA, rows, cols, vectorX);

	printf("i am currently here!!!\n");


	//printer(tempmatvec, rows1);



	*vectorR = residual(vectorb,tempmatvec,rows1);

	*vectorP = residual(vectorb,tempmatvec,rows1);

	

	

	normb = vecNorm(vectorb,rows1);

	rho = vecVec(vectorR, vectorR, rows1);

	if ( normb < EPSILON)
	{
		normb = 1.00;
	}

	while( vecNorm(vectorR,rows1) / normb > EPSILON )
	{
		*tempvec = matVec(matrixA,rows,cols,vectorP);

		store = vecVec(tempvec, vectorP, rows1);

		alpha = rho/ store ;

		*tempscalarvec = scalarVec(vectorP, alpha, rows1);

		*vectorX = vecAdd(vectorX, tempscalarvec, rows1);

		*tempscalarvec = scalarVec(tempvec, alpha, rows1);

		*vectorR = vecSub(vectorR, tempscalarvec, rows1);

		beta = vecVec(vectorR,vectorR,rows1 );

		tempstore1 = beta/rho ;

		*tempscalarvec = scalarVec(vectorP, tempstore1, rows1);

		*vectorP = vecAdd(vectorR, tempscalarvec, rows1);

		rho = beta;

		niter++;

		if (niter == maxiter)
		{
			break;

		}       


	
	}



	free(matrixA);
	free(vectorb);
	free(vectorX);
	free(vectorR);
	free(vectorP);
	free(tempvec);

	free(tempscalarvec);

	free(tempmatvec);



	return 0;

}

void initializeMatrixA(float *matrix, int rows, int cols, FILE *reader)
{
	int i,j;

	reader = fopen("matrixA.txt", "r");

	if(reader != NULL)
	{
		for(i=0; i<rows; i++)
		{
			for(j=0; j<cols; j++)
			{
				fscanf(reader,"%f%*c",&matrix[i*cols+j]);
			}
		}

		fclose(reader);

	}

	else
	{
		printf("Could not open file\n");
	}
	
	
}


void initializeVectB(float *vect, int rows1, FILE *reader)
{

	int i;

	reader = fopen("vectorb.txt", "r");

	if(reader != NULL)
	{
		for(i=0; i<rows1; i++)
		{
				fscanf(reader,"%f%*c",&vect[i]);
			
		}

		fclose(reader);

	}

	else
	{
		printf("Could not open filejjjjj\n");
	}
	

}

void initializeGuessVec(float *vect, int rows1, FILE *reader)
{

	int i;

	reader = fopen("initialguess.txt", "r");

	if(reader != NULL)
	{
		for(i=0; i<rows1; i++)
		{
				fscanf(reader,"%f%*c",&vect[i]);
			
		}

		fclose(reader);

	}

	else
	{
		printf("Could not open file\n");
	}
	

}

float matVec(float *matrix, int rows, int cols, float *vectr)
{

	int i,j;

	float *temp;

	temp = malloc (cols*sizeof(float) );

	printf("%d\n", cols);


	for(i=0; i<rows; i++)
	{
		temp[i] = 0.0;

		for(j=0; j<cols; j++)
		{
			temp[i] += matrix[i*cols+j] * vectr[j];
		}

		
	}


	return *temp;

	

}


float residual(float *vectb, float *vectx, int rows1)
{
	int i;

	float *temp;

	temp = malloc (rows1*sizeof(float) );

	
	for(i=0; i<rows1; i++)
	{
		

		temp[i] = vectb[i] - vectx[i];
	}

	return *temp;

	free(temp);
	
}

float vecNorm(float *vectr, int rows1)
{
	int i;

	float temp;

	temp = 0.0;

	for(i=0; i<rows1; i++)
	{
		temp += vectr[i] * vectr[i];
	}

	return sqrt(temp);
}


float vecVec(float *vect1,float *vect2, int rows1)
{
	int i;

	float *temp;

	temp = malloc (rows1*sizeof(float) );

	temp[0] = 0.0;

	for (i=0; i < rows1; i++)
	{
		temp[i] += vect1[i] * vect2[i] ;
	}

	return *temp;

	free(temp);
}

float vecAdd(float *vect, float *vect1, int rows1)
{
	int i;

	float *temp;

	temp = malloc (rows1*sizeof(float) );

	for (i=0; i < rows1; i++)
	{
		temp[i] = vect[i] + vect1[i] ;
	}

	return *temp;

	free(temp);

}

float vecSub(float *vect, float *vect1, int rows1)
{
	int i;

	float *temp;

	temp = malloc (rows1*sizeof(float) );

	for (i=0; i < rows1; i++)
	{
		temp[i] = vect[i] - vect1[i] ;
	}

	return *temp;

	free(temp);

}


float scalarVec(float *vect, float a, int rows1)
{
	int i;
	float *temp;

	temp = malloc (rows1*sizeof(float) );

	for(i=0; i<rows1; i++)
	{
		temp[i] = vect[i] * a;
	}

	return *temp;

	free(temp);

}

void printer(float *vect, int rows1)
{
	int i;

	for(i=0; i<rows1; i++)
	{
		printf("%f\n", vect[i]);

		
	}

}
