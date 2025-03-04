package iialib.games.model;

import java.util.ArrayList;

/**
 * Used to chararacterize the boards.he
 * It has to be impemented by some class in a real game.
 *
 * @param <Move>  Class implementing the moves for the game
 * @param <Role>  Class implementing the roles for the game
 * @param <Board> Class implementing the boards for the game
 */
public interface IBoard<Move extends IMove, Role extends IRole, Board extends IBoard<Move, Role, Board>> {

    /**
     * returns the possible moves a player having the playerRole
     *
     * @param playerRole
     * @return a list of all possible moves for the player having the playerRole
     */
    ArrayList<Move> possibleMoves(Role playerRole);

    /**
     * play move on the board, played by a player having the playerRole
     *
     * @param move
     * @param playerRole
     * @return the successor board
     */
    Board play(Move move, Role playerRole);

    /**
     * checks that move is valid for the player having the playerRole
     *
     * @param move
     * @param playerRole
     * @return yes if the move is valid for  playerRole
     */
    boolean isValidMove(Move move, Role playerRole);

    /**
     * checks that the board corresponds to an end of game
     *
     * @return yes if the game completed
     */
    boolean isGameOver();


    /**
     * returns the scores for each role (when the game is over)
     *
     * @return
     */
    ArrayList<Score<Role>> getScores();

}
