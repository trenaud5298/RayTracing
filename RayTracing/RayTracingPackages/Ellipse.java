package RayTracingPackages;

public class Ellipse {
    //ELLIPSE FORM : (x-x1)**2      (y-y1)**2     (z-z1)**2
    //               ---------  +   ---------  +  ---------  =   r**2
    //                 a**2           b**2          c**2
    //   x1 = centerPosX    y1 = centerPosY    z1 = centerPosZ
    //     r = radius
    //    a = widthScale      b = heightScale   z = lenghtScale
    //   a,b and c scale the dimensions of an otherwise perfect sphere with radius r
    //   i.e.  if a = 4, and r =3 then instead of just width 3 (r) from center to edge in x
    //   direction instead it would be (4*3) (a*r) to get a width of 12 from center to edge

    private double posX;        //x1
    private double posY;        //y1
    private double posZ;        //z1
    private double widthScale;  //a
    private double heightScale; //b
    private double lengthScale; //c
    private double radius;      //r
    private World worldSpace;
    private Color surfaceColor;
    private double reflectiveness;

    public Ellipse(World worldSpace, double posX, double posY, double posZ, double widthScale, double heightScale, double lengthScale, double radius, String colorName) {
        this.posX = posX;
        this.posY = posY;
        this.posZ = posZ;
        this.widthScale = widthScale;
        this.heightScale = heightScale;
        this.lengthScale = lengthScale;
        this.radius = radius;
        this.surfaceColor = new Color(colorName);
        this.worldSpace = worldSpace;
        this.worldSpace.addEllipse(this);
        this.reflectiveness = 0;
    }

   //Alteration Of Values
   public void incrimentX (double incriment) {
    this.posX += incriment;
    }

    public void incrimentY (double incriment) {
        this.posY += incriment;
    }

    public void incrimentZ (double incriment) {
        this.posZ += incriment;
    }

    public void incrimentRadius (double incriment) {
        this.radius += incriment;
    }

    public void incrimentWidth (double incriment) {
        this.widthScale += incriment;
    }

    public void incrimentHeight (double incriment) {
        this.heightScale += incriment;
    }

    public void incrimentLength (double incriment) {
        this.lengthScale += incriment;
    }


    //Variable Modification
    public void setX (double newX) {
        this.posX = newX;
    }

    public void setY (double newY) {
        this.posY = newY;
    }

    public void setZ (double newZ) {
        this.posZ = newZ;
    }

    public void setRadius (double newRadius) {
        this.radius = newRadius;
    }

    public void setWidth(double newWidth) {
        this.widthScale = newWidth;
    }

    public void setHeight(short newHeight) {
        this.heightScale = newHeight;
    }

    public void setLength(short newLength) {
        this.lengthScale = newLength;
    }

    public void setColor(int colorR, int colorG, int colorB) {
        this.surfaceColor.setColorValue(colorR,colorG,colorB);
    }

    public void setReflectiveness(double reflectiveness) {
        this.reflectiveness = reflectiveness;
    }


    //Variiable Retrieval
    public double getX (){
        return this.posX;
    }

    public double getY (){
        return this.posY;
    }

    public double getZ (){
        return this.posZ;
    }

    public double getRadius () {
        return this.radius;
    }

    public double getWidth () {
        return this.widthScale;
    }

    public double getHeight () {
        return this.heightScale;
    }

    public double getLength () {
        return this.lengthScale;
    }

    public int getColorIntValue() {
        return this.surfaceColor.getColorIntValue();
    }

    public double getReflectiveness() {
        return this.reflectiveness;
    }
}


