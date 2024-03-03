package RayTracingPackages;


public class Sphere {

    private double posX;
    private double posY;
    private double posZ;
    private double radius;
    private Color surfaceColor;
    private float reflectiveness;
    private World worldSpace;

    public Sphere (World worldSpace, double posX, double posY, double posZ, double radius, short colorR, short colorG, short colorB, float reflectiveness){
        this.posX = posX;
        this.posY = posY;
        this.posZ = posZ;
        this.radius = radius;
        this.surfaceColor = new Color(colorR,colorG,colorB);
        this.reflectiveness = reflectiveness;
        this.worldSpace = worldSpace;
        this.worldSpace.addSphere(this);
    }

    //Alteration Of Values
    public void translateX (double translation) {
        this.posX += translation;
    }
    
    public void translateY (double translation) {
        this.posY += translation;
    }
    
    public void translateZ (double translation) {
        this.posZ += translation;
    }

    public void changeRadius (double incriment) {
        this.radius += incriment;
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

    public void setColor(int colorR, int colorG, int colorB) {
        this.surfaceColor.setColorValue(colorR,colorG,colorB);
    }

    public void setReflectiveness(float newReflectiveness) {
        this.reflectiveness = newReflectiveness;
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

    public int getColorIntValue() {
        return this.surfaceColor.getColorIntValue();
    }

    public float getReflectiveness () {
        return this.reflectiveness;
    }

}



