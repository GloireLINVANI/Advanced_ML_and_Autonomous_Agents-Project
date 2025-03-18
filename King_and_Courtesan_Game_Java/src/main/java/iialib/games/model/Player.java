package iialib.games.model;


/**
 * used to associate an real player identifier to a role
 *
 * @param <Role>
 */


public class Player<Role extends IRole> {

    // ----------- Attributes  ------------

    /**
     * the role of the player in the game
     */
    Role role;
    /**
     * An (optional) identifier characterizing the player having that role in the game
     * This is useful for instance in a tournament, for keeping tra
     */
    String id;

    // ----------- Getters / Setters  ------------

    public Player(Role role) {
        this.role = role;
        this.id = "";
    }

    public Player(Role role, String id) {
        this(role);
        this.id = id;
    }

    public String getId() {
        return id;
    }
    // ----------- Constructors ------------

    public Role getRole() {
        return role;
    }

    public void setRole(Role role) {
        this.role = role;
    }

    // ----------- Other Methods ------------
    public String toString() {
        return role + (id.isEmpty() ? "" : " (" + id + ")");
    }


}
