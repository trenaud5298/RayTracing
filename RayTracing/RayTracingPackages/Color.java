package RayTracingPackages;

import java.util.Map;
import java.util.HashMap;

public class Color {
    private int colorIntValue;


    //PRE RECORDED COLORS
    //region#
    private static final Map<String,Integer> predefinedColors = new HashMap<>();
    static {
    predefinedColors.put("black",0);
    predefinedColors.put("white",16777215);
    predefinedColors.put("red",16711680);
    predefinedColors.put("green",65280);
    predefinedColors.put("blue",255);
    predefinedColors.put("yellow",16776960);
    predefinedColors.put("cyan",65535);
    predefinedColors.put("magenta",16711935);
    predefinedColors.put("gray",8421504);
    predefinedColors.put("dark gray",4210752);
    predefinedColors.put("light gray",12632256);
    predefinedColors.put("maroon",8388608);
    predefinedColors.put("olive",8421376);
    predefinedColors.put("navy",128);
    predefinedColors.put("purple",8388736);
    predefinedColors.put("teal",32896);
    predefinedColors.put("silver",12632256);
    predefinedColors.put("dark red",9109504);
    predefinedColors.put("brown",10824234);
    predefinedColors.put("firebrick",11674146);
    predefinedColors.put("crimson",14423100);
    predefinedColors.put("indian red",13458524);
    predefinedColors.put("light coral",15761536);
    predefinedColors.put("salmon",16416882);
    predefinedColors.put("dark salmon",15308410);
    predefinedColors.put("light salmon",16752762);
    predefinedColors.put("orange red",16729344);
    predefinedColors.put("dark orange",16747520);
    predefinedColors.put("orange",16753920);
    predefinedColors.put("gold",16766720);
    predefinedColors.put("dark goldenrod",12092939);
    predefinedColors.put("goldenrod",14329120);
    predefinedColors.put("pale goldenrod",15657130);
    predefinedColors.put("dark khaki",12433259);
    predefinedColors.put("khaki",15787660);
    predefinedColors.put("olive drab",7048739);
    predefinedColors.put("yellow green",10145074);
    predefinedColors.put("dark olive green",5597999);
    predefinedColors.put("olive green",7048739);
    predefinedColors.put("dark green",25600);
    predefinedColors.put("green yellow",11403055);
    predefinedColors.put("chartreuse",8388352);
    predefinedColors.put("lime",65280);
    predefinedColors.put("lime green",3329330);
    predefinedColors.put("forest green",2263842);
    predefinedColors.put("medium spring green",64154);
    predefinedColors.put("spring green",65407);
    predefinedColors.put("medium aquamarine",6737322);
    predefinedColors.put("medium sea green",3978097);
    predefinedColors.put("light sea green",2142890);
    predefinedColors.put("dark slate gray",3100495);
    predefinedColors.put("dark cyan",35723);
    predefinedColors.put("aqua",65535);
    predefinedColors.put("light cyan",14745599);
    predefinedColors.put("dark turquoise",52945);
    predefinedColors.put("turquoise",4251856);
    predefinedColors.put("medium turquoise",4772300);
    predefinedColors.put("pale turquoise",11529966);
    predefinedColors.put("aquamarine",8388564);
    predefinedColors.put("powder blue",11591910);
    predefinedColors.put("cadet blue",6266528);
    predefinedColors.put("steel blue",4620980);
    predefinedColors.put("cornflower blue",6591981);
    predefinedColors.put("deep sky blue",49151);
    predefinedColors.put("dodger blue",2003199);
    predefinedColors.put("light blue",11393254);
    predefinedColors.put("sky blue",8900331);
    predefinedColors.put("light sky blue",8900346);
    predefinedColors.put("midnight blue",1644912);
    predefinedColors.put("dark blue",139);
    predefinedColors.put("medium blue",205);
    predefinedColors.put("royal blue",4286945);
    predefinedColors.put("blue violet",9055202);
    predefinedColors.put("indigo",4915330);
    predefinedColors.put("dark slate blue",4734347);
    predefinedColors.put("slate blue",6970061);
    predefinedColors.put("medium slate blue",8087790);
    predefinedColors.put("medium purple",9662683);
    predefinedColors.put("dark orchid",10040012);
    predefinedColors.put("dark violet",9699539);
    predefinedColors.put("medium orchid",12211667);
    predefinedColors.put("thistle",14204888);
    predefinedColors.put("plum",14524637);
    predefinedColors.put("violet",15631086);
    predefinedColors.put("fuchsia",16711935);
    predefinedColors.put("orchid",14315734);
    predefinedColors.put("medium violet red",13047173);
    predefinedColors.put("pale violet red",14381203);
    predefinedColors.put("deep pink",16716947);
    predefinedColors.put("hot pink",16738740);
    predefinedColors.put("light pink",16758465);
    predefinedColors.put("pink",16761035);
    predefinedColors.put("antique white",16444375);
    predefinedColors.put("beige",16119260);
    predefinedColors.put("bisque",16770244);
    predefinedColors.put("blanched almond",16772045);
    predefinedColors.put("wheat",16113331);
    predefinedColors.put("cornsilk",16775388);
    predefinedColors.put("lemon chiffon",16775885);
    predefinedColors.put("light goldenrod yellow",16448210);
    predefinedColors.put("light yellow",16777184);
    predefinedColors.put("saddle brown",9127187);
    predefinedColors.put("sienna",10506797);
    predefinedColors.put("chocolate",13789470);
    predefinedColors.put("peru",13468991);
    predefinedColors.put("sandy brown",16032864);
    predefinedColors.put("burlywood",14596231);
    predefinedColors.put("tan",13808780);
    predefinedColors.put("rosy brown",12357519);
    predefinedColors.put("moccasin",16770229);
    predefinedColors.put("navajo white",16768685);
    predefinedColors.put("peach puff",16767673);
    predefinedColors.put("misty rose",16770273);
    predefinedColors.put("lavender blush",16773365);
    predefinedColors.put("linen",16445670);
    predefinedColors.put("old lace",16643558);
    predefinedColors.put("papaya whip",16773077);
    predefinedColors.put("sea shell",16774638);
    predefinedColors.put("mint cream",16121850);
    predefinedColors.put("slate gray",7372944);
    predefinedColors.put("light slate gray",7833753);
    predefinedColors.put("light steel blue",11584734);
    predefinedColors.put("lavender",15132410);
    predefinedColors.put("floral white",16775920);
    predefinedColors.put("alice blue",15792383);
    predefinedColors.put("ghost white",16316671);
    predefinedColors.put("honeydew",15794160);
    predefinedColors.put("ivory",16777200);
    predefinedColors.put("azure",15794175);
    predefinedColors.put("snow",16775930);
    predefinedColors.put("dim gray",6908265);
    predefinedColors.put("dark sea green",9419919);
    predefinedColors.put("navy blue",128);
    predefinedColors.put("golden brown",14329120);
    predefinedColors.put("dark golden brown",12092939);
    }
    //#endregion

    public Color (int colorR, int colorG, int colorB) {
        this.colorIntValue = (colorR << 16) | (colorG << 8) | colorB;
    }

    public Color (String colorName) {
        if (predefinedColors.containsKey(colorName.toLowerCase())) {
            this.colorIntValue = predefinedColors.get(colorName.toLowerCase());
        } else {
            System.out.println("ERROR: COLOR NAME NOT RECOGNIZED; USING DEFAULT VALUE");
            this.colorIntValue = 0;
        }
    }

    public int getColorIntValue() {
        return this.colorIntValue;
    }

    public void setColorValue(int colorR, int colorG, int colorB) {
        this.colorIntValue = (colorR << 16) | (colorG << 8) | colorB;
    }

    public void setColor(String colorName) {
        if (predefinedColors.containsKey(colorName.toLowerCase())) {
            this.colorIntValue = predefinedColors.get(colorName.toLowerCase());
        } else {
            System.out.println("ERROR: COLOR NAME NOT RECOGNIZED; COLOR NOT CHANGED");
        }
    }

}