package RayTracingPackages;

public class Plane {
    //Plane FORM :    a(x-x1)      b(y-y1)     c(z-z1)    =    d
    //
    //
    //   x1 = centerPosX    y1 = centerPosY    z1 = centerPosZ (JUST SHIFTS THE ax + by + cz = d plane around)
    //   <a,b,c> = normal vector to the plane
    //   d is a value thay typically controls height however it will always equal zero for us
    //   since we use x1,y1 and z1 to create a center point on the plane
    
    private Vector normalVector;
    private double a;
    private double b;
    private double c;
    private double d = 0;
    private double x1;
    private double y1;
    private double z1;
    private World worldSpace;
    private Color surfaceColor;
    private double lowerXBounds;
    private double upperXBounds;
    private double lowerYBounds;
    private double upperYBounds;
    private double lowerZBounds;
    private double upperZBounds;
    private double width;
    private double height;
    private double length;



    //     this.lowerXBounds = (double)this.x1-(width/2.0);
    //     this.upperXBounds = (double)this.x1+(width/2.0);
    //     this.lowerYBounds = (double)this.y1-(height/2.0);
    //     this.upperYBounds = (double)this.y1+(height/2.0);
    //     this.lowerZBounds = (double)this.z1-(length/2.0);
    //     this.upperZBounds = (double)this.z1+(length/2.0);
    public Plane(World worldSpace, double centerPosX, double centerPosY, double centerPosZ, double width, double height, double length, double xAxisRotationAngle, double yAxisRotationAngle, double zAxisRotationAngle, int colorR, int colorG, int colorB) {
        this.x1 = centerPosX;
        this.y1 = centerPosY;
        this.z1 = centerPosZ;
        this.width = width;
        this.height = height;
        this.length = length;
        this.normalVector = new Vector(0,1,0,0,0,0);
        this.normalVector.rotateX(xAxisRotationAngle);
        this.normalVector.rotateY(yAxisRotationAngle);
        this.normalVector.rotateZ(zAxisRotationAngle);
        System.out.println(normalVector);
        this.a = this.normalVector.getDirectionX();
        this.b = this.normalVector.getDirectionY();
        this.c = this.normalVector.getDirectionZ();
        System.out.print(this.a + " ");
        System.out.print(this.b + " ");
        System.out.println(this.c + " ");
        this.worldSpace = worldSpace;
        this.worldSpace.addPlane(this);
        this.surfaceColor = new Color(colorR,colorG,colorB);
    }



    public double getA(){
        return this.a;
    }

    public double getB(){
        return this.b;
    }

    public double getC(){
        return this.c;
    }

    public double getD(){
        return this.d;
    }

    public double getX1(){
        return this.x1;
    }

    public double getY1(){
        return this.y1;
    }

    public double getZ1(){
        return this.z1;
    }

    public int getColorIntValue() {
        return this.surfaceColor.getColorIntValue();
    }

    public double getLowerXBounds(){
        return (this.x1 - (double)(this.width/2.0));
    }

    public double getUpperXBounds(){
        return (this.x1 + (double)(this.width/2.0));
    }

    public double getLowerYBounds(){
        return (this.y1 - (double)(this.height/2.0));
    }

    public double getUpperYBounds(){
        return (this.y1 + (double)(this.height/2.0));
    }

    public double getLowerZBounds(){
        return (this.z1 - (double)(this.length/2.0));
    }

    public double getUpperZBounds(){
        return (this.z1 + (double)(this.length/2.0));
    }

}