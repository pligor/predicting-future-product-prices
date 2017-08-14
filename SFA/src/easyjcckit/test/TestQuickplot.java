package easyjcckit.test;

import static easyjcckit.QuickPlot.*;

import java.util.Arrays;

import junit.framework.TestCase;

public class TestQuickplot extends TestCase {
	public void testplot() throws Exception {
		plot( new double[]{0,1,2,3,4,5}, new double[]{0,1,4,9,16,25});
		Thread.sleep(2000);
		double[] xaxis = new double[10];
		double[] values = new double[10];
		for( int i = 1; i <= 10; i++ ) {
			xaxis[i-1]= i;
			values[i-1] = Math.log(i);
		}
		System.out.println(Arrays.toString(xaxis));
		System.out.println(Arrays.toString(values));
		plot( xaxis, values);
		Thread.sleep(2000);
		addPlot( new double[]{0,1,2,3,4,5}, new double[]{0,1,4,9,16,25});
		addScatter( new double[]{0,1,2,3,4,5}, new double[]{2,3,5,10,17,15});
		Thread.sleep(5000);
	}
	public void test() throws Exception {
		int N = 10000;
		double[] x = new double[N];
		double[] y = new double[N];
		for( int i = 0; i < N; i++ ) {
			x[i] = i;
			y[i] = i * i;
		}
		line(x,y);
		Thread.sleep(5000);
	}
	public void testwait() throws Exception {
		int N = 10000;
		double[] x = new double[N];
		double[] y = new double[N];
		for( int i = 0; i < N; i++ ) {
			x[i] = i;
			y[i] = i * i;
		}
		line(x,y);
		waitGraphs();
	}
}
