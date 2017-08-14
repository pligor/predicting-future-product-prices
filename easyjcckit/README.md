EasyJCCKit
==========

This is a fork of JCCKit, http://jcckit.sourceforge.net , which had not been updated since 2004.

I've added a really easy to use plot function.  It's easy to add several plots 
to the same axes too.

How to use
==========

    import static easyjcckit.QuickPlot.*;

    double[] xaxis = new double[]{0,1,2,3,4,5};
    double[] yvalues = new double[]{0,1,4,9,16,25};
    plot( xaxis, yvalues ); // create a plot using xaxis and yvalues
    double[] yvalues = new double[]{0,1,2,3,4,5};
    addPlot( xaxis, yvalues ); // create a second plot on top of first

    System.out.println("Press enter to exit");
    System.in.read();

Scatter plots:

    import static easyjcckit.QuickPlot.*;

    double[] xaxis = new double[]{0,1,2,3,4,5};
    double[] yvalues = new double[]{0,1,4,9,16,25};
    scatter( xaxis, yvalues ); // create a plot using xaxis and yvalues
    double[] yvalues = new double[]{0,1,2,3,4,5};
    addScatter( xaxis, yvalues ); // create a second plot on top of first

    System.out.println("Press enter to exit");
    System.in.read();

Lines (with no symbols):


    import static easyjcckit.QuickPlot.*;

    double[] xaxis = new double[]{0,1,2,3,4,5};
    double[] yvalues = new double[]{0,1,4,9,16,25};
    line( xaxis, yvalues ); // create a plot using xaxis and yvalues
    double[] yvalues = new double[]{0,1,2,3,4,5};
    addLine( xaxis, yvalues ); // create a second plot on top of first

    System.out.println("Press enter to exit");
    System.in.read();

You can freely mix these plots plots.

Advanced usage
==============

See http://jcckit.sourceforge.net.  Note that 'jcckit' has been replaced by
'easyjcckit' in the package names.

