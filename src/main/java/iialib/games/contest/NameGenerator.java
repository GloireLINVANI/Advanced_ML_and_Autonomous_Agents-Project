package iialib.games.contest;

import java.util.Random;

public final class NameGenerator {

    static final String[] names = { "Bugs Bunny", "Daffy Duck", "Porky Pig", "Elmer Fudd", "Tweety", "Sylvester", "Road Runner",
            "Wile E. Coyote", "The Tasmanian Devil", "Yosemite Sam", "Pepe Le Pew", "Marvin the Martian",
            "Foghorn Leghorn", "Speedy Gonzales", "Bosko", "Buddy", "Egghead", "Sniffles", "Cecil Turtle",
            "Mac 'n Tosh", "The Three Bears", "Henery Hawk", "Beaky Buzzard", "Witch Hazel", "Gossamer", "Cool Cat",
            "Merlin the Magic Mouse" };

    public static String pickName(){
        return names[new Random().nextInt(names.length)];
    }

}
