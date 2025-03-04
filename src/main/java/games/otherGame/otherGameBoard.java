package games.otherGame;

import iialib.games.model.IBoard;
import iialib.games.model.Score;

import java.util.ArrayList;

public class otherGameBoard implements IBoard<otherGameMove, otherGameRole, otherGameBoard> {


    // ---------------------- Attributes ---------------------
    // Attributes

    //TODO

    // --------------------- IBoard Methods ---------------------

    @Override
    public ArrayList<otherGameMove> possibleMoves(otherGameRole playerRole) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public otherGameBoard play(otherGameMove move, otherGameRole playerRole) {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public boolean isValidMove(otherGameMove move, otherGameRole playerRole) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public boolean isGameOver() {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public ArrayList<Score<otherGameRole>> getScores() {
        // TODO Auto-generated method stub
        return null;
    }

    // --------------------- Other Methods ---------------------


}
