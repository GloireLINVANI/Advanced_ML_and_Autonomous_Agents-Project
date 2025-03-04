package iialib.games.model;

/**
 * class used to describe the score corresponding to each player role when the game is over
 */
public class Score<Role extends IRole> {

    /**
     *
     */
    private Role role;

    ;

    // ----------- Attributes ------------
    /**
     *
     */
    private Status status;
    /**
     * score can be just 1/0 or a real score depending on the game
     */
    private int score;

    public Score(Role role, Status status, int score) {
        super();
        this.role = role;
        this.status = status;
        this.score = score;
    }

    // ----------- Constructors ------------

    public Role getRole() {
        return role;
    }

    // ----------- Getter / Setters  ------------

    public Status getStatus() {
        return status;
    }

    public int getScore() {
        return score;
    }

    public String toString() {
        return "Score <" + role + "," + status + "," + score + ">";
    }

    // ----------- Other public methods  ------------

    /**
     *
     */
    public enum Status {WIN, LOOSE, TIE}

}