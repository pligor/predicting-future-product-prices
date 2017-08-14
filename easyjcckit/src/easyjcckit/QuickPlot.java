package easyjcckit;

import java.awt.event.*;
import java.util.*;
import java.util.concurrent.*;

import javax.swing.*;

import easyjcckit.data.*;
import easyjcckit.util.*;

/**
 * You can import this statically "import static easyjcckit.QuickPlot.*;
 */
public class QuickPlot {
	static JFrame jframe;

	public static class CurveData {
		public double[] xaxis;
		public double[] yvalues;
		public boolean line = true;
		public boolean symbol = true;
		public CurveData(double[] xaxis, double[] yvalues) {
			this.xaxis = xaxis;
			this.yvalues = yvalues;
		}
		public CurveData(double[] xaxis, double[] yvalues, boolean line) {
			this.xaxis = xaxis;
			this.yvalues = yvalues;
			this.line = line;
		}
		public CurveData symbol(boolean symbol){
			this.symbol = symbol;
			return this;
		}
	}

	static final ArrayList<CurveData> curves = new ArrayList<CurveData>();

	static void _plot() {
		double xmin, xmax, ymin, ymax;
		xmin = xmax = ymin = ymax = 0;
		for( int set = 0; set < curves.size(); set++ ) {
			CurveData curveData = curves.get(set);
			int N = curveData.xaxis.length;
			if( curveData.yvalues.length != N ) {
				throw new RuntimeException("xaxis and yvalues should have same length");
			}
			if( N == 0 ) {
				throw new RuntimeException("xaxis and yvalues were empty");
			}
			if( set == 0 ) {
				xmin = xmax = curveData.xaxis[0];
				ymin = ymax = curveData.yvalues[0];
			}
			for( int i = 1; i < N; i++ ) {
				xmin = Math.min(xmin, curveData.xaxis[i]);
				xmax = Math.max(xmax, curveData.xaxis[i]);
				ymin = Math.min(ymin, curveData.yvalues[i]);
				ymax = Math.max(ymax, curveData.yvalues[i]);
			}
			if( xmin == xmax ) {
				throw new RuntimeException("xaxis is singular");
			}
		}
		if( ymin == ymax ) {
			if( ymin > 0 ) {
				ymin = 0;
			} else if( ymin < 0 ) {
				ymax = 0;
			} else {
				ymin -= 1;
				ymax +=1 ;				
			}
		}

		Properties props = new Properties();
		ConfigParameters config = new ConfigParameters(new PropertiesBasedConfigData(props));
		props.put("foreground", "0");
		props.put("background", "0xffffff");
		props.put("paper", "0 0 1 1");
		props.put("horizontalAnchor", "left");
		props.put("verticalAnchor", "bottom");
		props.put("plot/legendVisible", "false");
		props.put("plot/coordinateSystem/xAxis/minimum", "" + xmin);
		props.put("plot/coordinateSystem/xAxis/maximum", "" + xmax);
		props.put("plot/coordinateSystem/xAxis/ticLabelFormat", "%d");
		props.put("plot/coordinateSystem/yAxis/axisLabel", "y");
		props.put("plot/coordinateSystem/yAxis/minimum", "" + ymin);
		props.put("plot/coordinateSystem/yAxis/maximum", "" + ymax);
		props.put("plot/coordinateSystem/yAxis/axisLength", "0.8");
		props.put("plot/coordinateSystem/xAxis/axisLength", "1.15");
		props.put("plot/coordinateSystem/yAxis/ticLabelFormat", "%d");
		String definitions = "";
		for( int set = 0; set < curves.size(); set++ ) {
			if( set != 0 ) {
				definitions += " ";
			}
			definitions += "y" + set;
		}
		String[] colors = new String[]{"0xff0000","0x00ff00","0x0000ff","0xffff00","0xff00ff","0x00ffff"};
		props.put("plot/curveFactory/definitions", definitions);
		for( int set = 0; set < curves.size(); set++ ) {
			if( curves.get(set).line ) {
				props.put("plot/curveFactory/y" + set + "/withLine", "true");
			} else {
				props.put("plot/curveFactory/y" + set + "/withLine", "false");
			}
			if( curves.get(set).symbol ) {
				props.put("plot/curveFactory/y" + set + "/symbolFactory/className", 
						"easyjcckit.plot.CircleSymbolFactory");
				props.put("plot/curveFactory/y" + set + "/symbolFactory/size", "0.01");
				props.put("plot/curveFactory/y" + set + "/symbolFactory/attributes/className",
						"easyjcckit.graphic.ShapeAttributes");
				props.put("plot/curveFactory/y" + set + "/symbolFactory/attributes/fillColor", colors[set % colors.length]);
				props.put("plot/curveFactory/y" + set + "/lineAttributes/className",
						"easyjcckit.graphic.ShapeAttributes");
				props.put("plot/curveFactory/y" + set + "/lineAttributes/lineColor", colors[set % colors.length]);
			}
		}

		final GraphicsPlotCanvas plotCanvas = new Graphics2DPlotCanvas(config);

		DataPlot _dataPlot = new DataPlot();
		for( int set = 0; set < curves.size(); set++ ) {
			CurveData curveData = curves.get(set);
			DataCurve curve = new DataCurve("y" + set);
			for( int i = 0; i < curveData.xaxis.length; i++ ) {
				curve.addElement(new DataPoint(curveData.xaxis[i],curveData.yvalues[i]));
			}
			_dataPlot.addElement(curve);
		}
		plotCanvas.connect(_dataPlot);

		if( jframe != null ) {
			jframe.setVisible(false);
			jframe.dispose();
			jframe = null;
		}
		jframe = new JFrame();
		jframe.setTitle("EasyJccKit");
		jframe.setSize(800,600);
		jframe.setLocationRelativeTo(null);
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jframe.add(plotCanvas.getGraphicsCanvas());
		jframe.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
		jframe.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent ev) {
				jframe.dispose();
			}
		});
		jframe.setVisible(true);
	}

	public static void scatter(double[] xaxis, double[] yvalues ) {
		curves.clear();
		curves.add(new CurveData(xaxis.clone(), yvalues.clone(), false));
		_plot();
	}

	public static void addScatter(double[] xaxis, double[] yvalues ) {
		curves.add(new CurveData(xaxis.clone(), yvalues.clone(), false));
		_plot();
	}

	public static void plot(double[] xaxis, double[] yvalues ) {
		curves.clear();
		curves.add(new CurveData(xaxis.clone(), yvalues.clone(), true));
		_plot();
	}

	public static void addPlot(double[] xaxis, double[] yvalues ) {
		curves.add(new CurveData(xaxis.clone(), yvalues.clone()));
		_plot();
	}
	public static void line(double[] xaxis, double[] yvalues ) {
		curves.clear();
		curves.add(new CurveData(xaxis.clone(), yvalues.clone(), true).symbol(false));
		_plot();
	}

	public static void addLine(double[] xaxis, double[] yvalues ) {
		curves.add(new CurveData(xaxis.clone(), yvalues.clone()).symbol(false));
		_plot();
	}
	public static void clearPlots() {
		curves.clear();		
		if( jframe != null ) {
			jframe.setVisible(false);
			jframe.dispose();
			jframe = null;
		}
	}
	public static void waitGraphs() {
		if( jframe == null ) {
			return;
		}
		while(jframe.isVisible() ) {
			try {
				Thread.sleep(100);
			} catch( Exception e ) {
				throw new RuntimeException("thread interrupted");
			}
		}
	}
}
